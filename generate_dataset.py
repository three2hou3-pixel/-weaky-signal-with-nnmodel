# ⚠ 注意：此代码用于生成模型训练的模拟数据，借用了 Github 上的开源项目 SeaClutterSuppression 来创建雷达数据和对应的掩膜。请确保你已经下载了该项目文件，并且有足够的磁盘空间来保存生成的数据集。
# 以下是该项目地址：https://github.com/pepijn-lens/SeaClutterSuppression.git

import h5py
import numpy as np
import sea_clutter
import random
import os
import sys
from typing import List
from src import generate_data

rp = sea_clutter.RadarParams(prf=5000.0, n_pulses=128, n_ranges=128, carrier_wavelength=0.03)
cp = sea_clutter.ClutterParams(mean_power_db=10.0, shape_param=0.7, wave_speed_mps=3.0)
sp = sea_clutter.SequenceParams(n_frames=50, frame_rate_hz=2.0)

n_sequences = 3000  # 3000个序列的 Tra

# 1. 定义数据保存的根目录
target_dir = '/weaky_signal/radar_data'

# 2. 使用 os.path.join 拼接出绝对路径，避免不同操作系统的斜杠问题
h5_filename = os.path.join(target_dir, 'radar_dataset_3000_50.h5')


def generate_and_save_h5():
    # ==================== 极度重要的防崩补丁 ====================
    # 3. 确保目标文件夹存在。如果不存在，代码会自动帮你逐层创建
    os.makedirs(target_dir, exist_ok=True)
    # ============================================================

    if os.path.exists(h5_filename):
        os.remove(h5_filename)
    
    print(f"开始批量生成：{n_sequences} 个序列, 每个序列 {sp.n_frames} 帧。")
    print(f"数据将安全地保存到数据盘：{h5_filename}")

    with h5py.File(h5_filename, 'w') as f:
        # 创建 RDM 和 Mask 的数据集
        rdm_dataset = f.create_dataset('rdm_data',
            shape=(n_sequences, sp.n_frames, rp.n_ranges, rp.n_pulses),
            dtype=np.float32,
            chunks=(1, sp.n_frames, rp.n_ranges, rp.n_pulses))
        
        mask_dataset = f.create_dataset('mask_data',
            shape=(n_sequences, sp.n_frames, rp.n_ranges, rp.n_pulses),
            dtype=np.uint8,
            chunks=(1, sp.n_frames, rp.n_ranges, rp.n_pulses))

        for s_idx in range(n_sequences):
            # 初始化 20 个目标
            targets = []
            for _ in range(20):
                tgt = sea_clutter.RealisticTarget(
                    rng_idx=random.randint(0, rp.n_ranges - 1),
                    doppler_hz=0.0,
                    power=10**(random.uniform(10.0, 14.0) / 10.0),
                    size=5
                )
                tgt.current_velocity_mps = random.uniform(-15.0, 15.0)
                targets.append(tgt)

            # 生成当前序列的所有帧和掩膜
            rdm_list, mask_list = generate_data.simulate_sequence_with_realistic_targets_and_masks(
                rp=rp, cp=cp, sp=sp, targets=targets
            )

            # 转换为 Numpy 数组并处理 RDM 为 dB 格式
            seq_rdm_array = np.array(rdm_list)
            seq_rdm_db = 10 * np.log10(np.abs(seq_rdm_array) + 1e-12).astype(np.float32)

            # Mask 转化为 uint8
            seq_mask_array = np.array(mask_list).astype(np.uint8)

            # 写入 HDF5 文件
            rdm_dataset[s_idx, :, :, :] = seq_rdm_db
            mask_dataset[s_idx, :, :, :] = seq_mask_array

            if (s_idx + 1) % 10 == 0:
                print(f"处理进度：{s_idx + 1}/{n_sequences} 个序列已保存")

if __name__ == "__main__":
    generate_and_save_h5()
    print("数据生成和保存完成！")