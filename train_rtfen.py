import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from scipy.ndimage import label, center_of_mass
import math

from functions import RTFEN, TemporalTverskyLoss

# ====================== 1. 路径与全局配置 ======================
DATA_PATH = '/root/autodl-tmp/radar_data/low_state_dataset/radar_dataset_3000_50.h5'
SAVE_DIR = '/root/autodl-tmp/RTFEN_outputs/checkpoints'
LOG_DIR = '/root/autodl-tmp/RTFEN_outputs/logs'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

BATCH_SIZE = 4
MAX_ITERS = 500      
PATIENCE = 50        
LR = 1e-4
THRESH = 0.5         
GRADIENT_CLIP = 1.0  

# ====================== 2. 数据加载器 (共享内存，彻底消灭 I/O 死锁) ======================
class RadarH5Dataset(Dataset):
    def __init__(self, rdm_tensor, mask_tensor):
        self.rdm_tensor = rdm_tensor
        self.mask_tensor = mask_tensor

    def __len__(self):
        return self.rdm_tensor.shape[0]

    def __getitem__(self, idx):
        x = self.rdm_tensor[idx]  
        y = self.mask_tensor[idx] 
        
        # 依据论文参数  推导：纯净低海况最大范围设为 40dB，安全等比例缩放
        x = torch.clamp(x, 0.0, 40.0) / 40.0

        x_tensor = x.float().unsqueeze(0)  # [1, T, H, W]
        y_tensor = y.float().unsqueeze(1)  # [T, 1, H, W]
        
        return x_tensor, y_tensor

# ====================== 3. 欧氏距离模糊指标评估 (CCA 质心分析版) ======================
def evaluate_frame_with_cca(pred_mask_numpy, true_mask_numpy, distance_threshold=1.5):
    """严格对齐雷达物理的连通域质心距离评估"""
    pred_labeled, pred_num_features = label(pred_mask_numpy, structure=np.ones((3,3)))
    true_labeled, true_num_features = label(true_mask_numpy, structure=np.ones((3,3)))
    
    pred_centroids = center_of_mass(pred_mask_numpy, pred_labeled, range(1, pred_num_features + 1))
    true_centroids = center_of_mass(true_mask_numpy, true_labeled, range(1, true_num_features + 1))
    
    pred_centroids = [c for c in pred_centroids if not np.isnan(c[0])]
    true_centroids = [c for c in true_centroids if not np.isnan(c[0])]

    TP, FP, FN = 0, 0, 0
    
    if len(true_centroids) == 0:
        return 0, len(pred_centroids), 0
    if len(pred_centroids) == 0:
        return 0, 0, len(true_centroids)

    matched_true_indices = set()
    for pred_c in pred_centroids:
        min_dist = float('inf')
        best_true_idx = -1
        
        for i, true_c in enumerate(true_centroids):
            dist = math.sqrt((pred_c[0] - true_c[0])**2 + (pred_c[1] - true_c[1])**2)
            if dist < min_dist:
                min_dist = dist
                best_true_idx = i
                
        if min_dist <= distance_threshold and best_true_idx not in matched_true_indices:
            TP += 1
            matched_true_indices.add(best_true_idx) 
        else:
            FP += 1 
            
    FN = len(true_centroids) - len(matched_true_indices)
    return TP, FP, FN

def evaluate(model, dataloader, criterion, device, thresh=0.5):
    model.eval()
    total_loss = 0.0
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # 转回 CPU 使用 CCA 评估，避免 GPU 阻塞
            preds_bin = (outputs > thresh).float().cpu().numpy()
            true_masks = targets.cpu().numpy()
            
            for b in range(preds_bin.shape[0]):
                for t in range(preds_bin.shape[1]):
                    tp, fp, fn = evaluate_frame_with_cca(preds_bin[b, t, 0, :, :], true_masks[b, t, 0, :, :], distance_threshold=1.5)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

    avg_loss = total_loss / len(dataloader)
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return avg_loss, f1_score, precision, recall

# ====================== 4. 主流程 ======================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 当前算力终端: {device} | 准备起飞...")

    # 🌟 核心突破：一次性读入共享内存，彻底解决 GPU 饿死和内存暴涨
    print("🔥 正在将数据集一次性读入系统共享内存通道...")
    with h5py.File(DATA_PATH, 'r') as f:
        rdm_all = torch.from_numpy(f['rdm_data'][:]).float()
        mask_all = torch.from_numpy(f['mask_data'][:]).float()
    print("✅ 数据成功常驻物理内存。正在构建 DataLoader...")

    full_dataset = RadarH5Dataset(rdm_all, mask_all)
    total_size = len(full_dataset)
    
    train_size = int(0.75 * total_size) # 严格遵循论文 7.5:1.5:1.5 [cite: 156]
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=2, pin_memory=True)

    model = RTFEN(in_channels=1, L=32, T=50, original_size=(128, 128)).to(device)
    criterion = TemporalTverskyLoss(alpha=0.2, beta=0.8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4) 
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=LOG_DIR)

    best_val_f1 = 0.0
    early_stop_counter = 0

    print(f"📈 开始 RTFEN 论文标准训练，最大迭代 {MAX_ITERS} 轮，Patience: {PATIENCE}") 
    
    for epoch in range(1, MAX_ITERS + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()  

        val_loss, val_f1, val_p, val_r = evaluate(model, val_loader, criterion, device, THRESH)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:03d}/{MAX_ITERS}] | LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1*100:.2f}% (P: {val_p*100:.2f}%, R: {val_r*100:.2f}%)")

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Metrics/F1_Score', val_f1, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"RTFEN_epoch_{epoch}.pth"))
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1  # 已修复原代码的双重赋值笔误
            early_stop_counter = 0 
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_RTFEN_model.pth"))
            print(f"   🎯 发现更优模型！当前最高 Val F1: {best_val_f1*100:.2f}%，已保存！")
        else:
            early_stop_counter += 1
            print(f"   ⏳ 早停计数器: {early_stop_counter} / {PATIENCE}")

        if early_stop_counter >= PATIENCE:
            print(f"🛑 触发早停机制！连续 {PATIENCE} 轮 Val F1 未提升。")
            break

    writer.close()
    
    # ================= 5. 最终测试阶段 =================
    print("\n================= 🏆 最终测试阶段 =================")
    best_model_path = os.path.join(SAVE_DIR, "best_RTFEN_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("✅ 已加载最优权重，正在处理测试集...")
        
        test_loss, test_f1, test_p, test_r = evaluate(model, test_loader, criterion, device, THRESH)
        
        print("\n" + "="*45)
        print("🎉 RTFEN 最终测试结果 🎉")
        print(f"   Test Loss:      {test_loss:.4f}")
        print(f"   Test Precision: {test_p * 100:.2f}%")
        print(f"   Test Recall:    {test_r * 100:.2f}%")
        print(f"   Test F1-Score:  {test_f1 * 100:.2f}%")
        print("="*45)
        
        with open(os.path.join(LOG_DIR, "final_test_results.txt"), "w") as f:
            f.write(f"Test Precision: {test_p:.4f}\n")
            f.write(f"Test Recall: {test_r:.4f}\n")
            f.write(f"Test F1-Score: {test_f1:.4f}\n")
            
        print("⏳ 训练完成，10秒后自动关机...")
        os.system("sleep 10 && shutdown") 

if __name__ == '__main__':
    main()