import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# 从你编写的 functions.py 中导入模型和损失函数
from functions import RTFEN, TemporalTverskyLoss

# ====================== 1. 路径与超参数配置 ======================
DATA_PATH = '/root/autodl-tmp/radar_data/toy_dataset/radar_dataset_300_toy.h5'
SAVE_DIR = '/root/autodl-tmp/rften_outputs/checkpoints'
LOG_DIR = '/root/autodl-tmp/rften_outputs/logs'

# 创建必要的文件夹
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 训练超参数
BATCH_SIZE = 4       # 300个序列较小，4 或 8 非常合适
EPOCHS = 60          # Toy Dataset 验证阶段 50-60 个 Epoch 足够看清趋势
LR = 1e-4            # 初始学习率
VAL_SPLIT = 0.2      # 验证集比例 (300 * 0.2 = 60 个验证序列)
THRESH = 0.5         # 验证集计算 F1-Score 的二值化阈值

# ====================== 2. 自定义数据集类 ======================
class RadarH5Dataset(Dataset):
    def __init__(self, rdm_data, mask_data):
        """
        rdm_data: Numpy array, 形状 [N, T, H, W]
        mask_data: Numpy array, 形状 [N, T, H, W]
        """
        self.rdm_data = rdm_data
        self.mask_data = mask_data

    def __len__(self):
        return self.rdm_data.shape[0]

    def __getitem__(self, idx):
        # 1. 提取单条序列数据
        x = self.rdm_data[idx]  # [T, H, W]
        y = self.mask_data[idx]  # [T, H, W]

        # 2. 转换为 PyTorch 张量
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).float()

        # 3. 维度调整以对接网络输入
        # 模型 forward 输入需要 [B, 1, T, H, W]，此处增加 Channel 维变 [1, T, H, W]
        x_tensor = x_tensor.unsqueeze(0) 
        # 损失函数需要 [B, T, 1, H, W]，此处增加 Channel 维变 [T, 1, H, W]
        y_tensor = y_tensor.unsqueeze(1)

        return x_tensor, y_tensor

def prepare_dataloaders(h5_path, val_split=0.2, batch_size=4):
    print("正在从数据盘加载 HDF5 数据到内存...")
    with h5py.File(h5_path, 'r') as f:
        rdm_all = f['rdm_data'][:]   # type: ignore # [300, 50, 128, 128]
        mask_all = f['mask_data'][:] # type: ignore # [300, 50, 128, 128]
    
    num_samples = rdm_all.shape[0] # type: ignore
    num_val = int(num_samples * val_split)
    num_train = num_samples - num_val

    # 划分训练集与验证集
    train_rdm, val_rdm = rdm_all[:num_train], rdm_all[num_train:] # type: ignore
    train_mask, val_mask = mask_all[:num_train], mask_all[num_train:] # type: ignore

    train_dataset = RadarH5Dataset(train_rdm, train_mask)
    val_dataset = RadarH5Dataset(val_rdm, val_mask)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"数据集划分完成 -> 训练集: {num_train} 个序列, 验证集: {num_val} 个序列")
    return train_loader, val_loader

# ====================== 3. 辅助评估函数 ======================
def calculate_metrics(preds, targets, thresh=0.5):
    """
    计算二分类的 TP, FP, FN 用于后期的 Precision, Recall, F1 计算
    preds, targets: 展平后的 1D 张量
    """
    preds_bin = (preds > thresh).float()
    
    tp = (preds_bin * targets).sum().item()
    fp = (preds_bin * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds_bin) * targets).sum().item()
    
    return tp, fp, fn

# ====================== 4. 主训练逻辑 ======================
def main():
    # 检测硬件
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用设备: {device}")

    # 加载数据
    train_loader, val_loader = prepare_dataloaders(DATA_PATH, val_split=VAL_SPLIT, batch_size=BATCH_SIZE)

    # 实例化网络与损失函数 (参数严格对应你的 functions.py 默认设置)
    model = RTFEN(in_channels=1, L=32, T=50, original_size=(128, 128)).to(device)
    criterion = TemporalTverskyLoss(alpha=0.2, beta=0.8) # 纯净版时空 Tversky 损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # 实例化 TensorBoard
    writer = SummaryWriter(log_dir=LOG_DIR)

    best_f1 = 0.0

    RESUME_EPOCH = 30  
    
    if RESUME_EPOCH > 0:
        checkpoint_path = os.path.join(SAVE_DIR, f"model_epoch_{RESUME_EPOCH}.pth")
        if os.path.exists(checkpoint_path):
            # 把硬盘里的权重加载到模型里
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"✅ 成功抢救回进度！已加载权重: {checkpoint_path}")
            print(f"🚀 将直接从第 {RESUME_EPOCH + 1} 个 Epoch 开始往下跑！")
        else:
            print(f"❌ 找不到权重文件 {checkpoint_path}，将从头开始跑。")
            RESUME_EPOCH = 0
    else:
        print("🌱 从头开始训练网络...")
        
    for epoch in range(1, EPOCHS + 1):
        # ------------------ 训练阶段 ------------------
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)   # [B, 1, T, H, W]
            targets = targets.to(device) # [B, T, 1, H, W]

            optimizer.zero_grad()
            outputs = model(inputs)      # [B, T, 1, H, W]
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ------------------ 验证阶段 ------------------
        model.eval()
        val_loss = 0.0
        total_tp, total_fp, total_fn = 0, 0, 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # 展平数据用于计算全局 F1-Score
                preds_flat = outputs.view(-1)
                targets_flat = targets.view(-1)
                
                tp, fp, fn = calculate_metrics(preds_flat, targets_flat, thresh=THRESH)
                total_tp += tp
                total_fp += fp
                total_fn += fn

        avg_val_loss = val_loss / len(val_loader)
        
        # 计算 Precision, Recall, F1 (加入 1e-6 防止分母为 0)
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

        # ------------------ 日志记录与打印 ------------------
        print(f"Epoch [{epoch:02d}/{EPOCHS}] "
              f"| Train Loss: {avg_train_loss:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} "
              f"| Val F1: {f1_score:.4f} (P: {precision:.4f}, R: {recall:.4f})")

        # 写入 TensorBoard 曲线
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Metrics/F1_Score', f1_score, epoch)
        writer.add_scalar('Metrics/Precision', precision, epoch)
        writer.add_scalar('Metrics/Recall', recall, epoch)

        # ------------------ 模型保存策略 ------------------
        # 1. 定期备份
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"model_epoch_{epoch}.pth"))
        
        # 2. 保存最优模型（以 F1-Score 为准）
        if f1_score > best_f1:
            best_f1 = f1_score
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"发现更好的模型，已保存至 best_model.pth，当前最高 F1: {best_f1:.4f}")

    writer.close()
    print(f"训练完成！最优模型的验证集 F1-Score 为: {best_f1:.4f}")

if __name__ == '__main__':
    main()