import os
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ================= 路径配置 =================
# 指向你的混合数据集
DATA_PATH = 'radar_dataset_30_50.h5'
# 创建一个专门用于可视化的 TensorBoard 目录
LOG_DIR = 'tensorboard_logs'

os.makedirs(LOG_DIR, exist_ok=True)

def visualize_to_tensorboard():
    print(f"正在打开数据集: {DATA_PATH}")
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    with h5py.File(DATA_PATH, 'r') as f:
        rdm_data = f['rdm_data']
        mask_data = f['mask_data']
        total_seqs = rdm_data.shape[0] # type: ignore
        
        # 随机抽取 5 个序列来观察
        sample_indices = random.sample(range(total_seqs), 5)
        
        for i, seq_idx in enumerate(sample_indices):
            # 获取该序列的全部 50 帧
            seq_rdm = rdm_data[seq_idx]   # type: ignore # [50, 128, 128]
            seq_mask = mask_data[seq_idx] # type: ignore # [50, 128, 128]
            
            # 我们不需要看全部 50 帧，等间距抽取 4 帧就够了
            frames_to_show = [0, 15, 30, 49]
            
            # 创建一个大画布：4 行 (对应 4 帧) x 2 列 (RDM 和 Mask)
            fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 16))
            fig.suptitle(f"Sequence {seq_idx} Visualization", fontsize=16)
            
            for row_idx, frame_idx in enumerate(frames_to_show):
                rdm_frame = seq_rdm[frame_idx] # type: ignore
                mask_frame = seq_mask[frame_idx] # type: ignore
                
                # ---- 画 RDM (雷达距离-多普勒图) ----
                ax_rdm = axes[row_idx, 0]
                # 使用 jet 或者 viridis 颜色映射，这最符合雷达图像的习惯
                im1 = ax_rdm.imshow(rdm_frame, cmap='jet', aspect='auto')
                ax_rdm.set_title(f"Frame {frame_idx} - RDM (dB)")
                fig.colorbar(im1, ax=ax_rdm, fraction=0.046, pad=0.04)
                
                # ---- 画 Mask (真实标签) ----
                ax_mask = axes[row_idx, 1]
                im2 = ax_mask.imshow(mask_frame, cmap='gray', aspect='auto', vmin=0, vmax=1)
                ax_mask.set_title(f"Frame {frame_idx} - Ground Truth Mask")
                
            plt.tight_layout()
            
            # 将这个完美的对比图发送到 TensorBoard
            writer.add_figure(f'Mixed_Data_Samples/Sequence_{seq_idx}', fig, global_step=0)
            print(f"✅ 序列 {seq_idx} 的图像已写入 TensorBoard")
            
    writer.close()
    print("\n🎉 全部写入完成！请打开 TensorBoard 查看。")

if __name__ == "__main__":
    visualize_to_tensorboard()