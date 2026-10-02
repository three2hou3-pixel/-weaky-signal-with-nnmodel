# 弱信号检测：RTFEN 雷达海杂波抑制网络

针对**低海况下的弱小目标检测**，复现并实现了一个端到端的时空特征增强网络 **RTFEN**（Radar Temporal Feature Enhancement Network）：输入连续 50 帧的距离-多普勒图（RDM），直接输出同样 50 帧的目标分割掩膜，逐帧给出目标位置。

> 论文复现性质。数据为仿真生成，不含任何实测或涉密数据。

## 网络结构

整体是一条 **5 段式流水线**，全部实现在 `functions.py` 中：

```
输入 [B, 1, T=50, H=128, W=128]   (50 帧 RDM 时序)
   │
   ├─ ① TemporalCompressionModule     3D 卷积 x2，通道 1 → L → 2L
   │                                   提取初级空时特征
   ├─ ② BiConvLSTM                    双向 ConvLSTM，前向 + 后向时序演化
   │                                   → [B, 4L, T, H, W]
   ├─ ③ TemporalAttentionFusion       时间注意力加权 + 多尺度融合
   │                                   压掉时间维 → [B, 4L, H, W]
   ├─ ④ UNet                          2 级编码器-解码器，跳跃连接
   │                                   → [B, 4L, H, W]
   └─ ⑤ MultiFramePredictionHead      T 个**参数独立**的 1x1 卷积头并行预测
                                       双线性上采样 + Sigmoid
输出 [B, T=50, 1, H=128, W=128]   逐帧目标概率图
```

**关键设计**：预测头不是共享权重，而是 `T=50` 个完全独立的 `1x1 Conv`（用 `nn.ModuleList` 包装），让每一帧都能学到自己的判别边界。

## 损失函数

`TemporalTverskyLoss(alpha=0.2, beta=0.8)` —— 把 Tversky Index 逐帧展开后对所有帧求平均：

$$\text{TI} = \frac{TP + \epsilon}{TP + \alpha \cdot FP + \beta \cdot FN + \epsilon}, \qquad \mathcal{L} = 1 - \text{TI}$$

`beta=0.8 > alpha=0.2` 意味着**对漏检（FN）的惩罚远大于误报（FP）**——弱小目标检测场景下，漏掉一个真实目标的代价比多报几个虚假目标高得多。

## 评估指标

不采用逐像素 IoU，而是使用**基于连通域质心（CCA）的欧氏距离匹配**（`train_rtfen.py: evaluate_frame_with_cca`）：对预测掩膜和真值掩膜分别做连通域分析、取质心，质心距离 ≤ 1.5 像素即判为命中。这更贴近雷达实际关注的「有没有检出这个目标」，而非「掩膜画得准不准」。

## 项目结构

```
weaky-signal/
├── functions.py            # RTFEN 全部模块 + TemporalTverskyLoss
├── generate_dataset.py     # 仿真数据集生成（依赖 SeaClutterSuppression）
├── train.py                # Toy 数据集快速验证（300 序列）
├── train_rtfen.py          # 完整训练脚本（3000 序列，含 CCA 评估 + 早停）
├── visualize_tb.py         # 把 RDM / Mask 对比图写入 TensorBoard
└── saved_models/           # 训练好的权重
```

## 快速开始

### 1. 生成仿真数据集

数据集本身**不随仓库提供**（体积过大），需要先生成。本项目的仿真数据依赖开源项目
[**SeaClutterSuppression**](https://github.com/pepijn-lens/SeaClutterSuppression)（作者 Pepijn Lens，Leiden University / TNO），请先自行克隆该仓库（见下方「致谢与第三方依赖」）。

```bash
python generate_dataset.py    # 生成 3000 序列 x 50 帧的 HDF5 数据集
```

### 2. 训练

```bash
python train.py          # 小规模快速验证（300 序列）
python train_rtfen.py    # 完整训练（3000 序列，75% / 15% / 15% 划分）
```

> **运行前需修改脚本中的数据路径**：`train.py` 与 `train_rtfen.py` 中的 `DATA_PATH` / `SAVE_DIR` / `LOG_DIR` 目前是作者在 AutoDL 云服务器上的绝对路径（`/root/autodl-tmp/...`），请改成你自己的本地路径。
>
> **关于自动关机**：`train_rtfen.py` 顶部有 `AUTO_SHUTDOWN = False` 开关。默认关闭，训练结束只打印提示；在云服务器上想省钱时可手动改为 `True`，训练结束后会自动执行 `shutdown`。

## 训练配置

| 项 | 值 |
|---|---|
| 输入 | 3000 序列 × 50 帧 × 128×128 的 RDM（dB 尺度，截断至 40dB 后归一化） |
| 划分 | 75% / 15% / 15%（train / val / test），seed=42 |
| 优化器 | AdamW (lr=1e-4, weight_decay=1e-4) |
| 调度 | CosineAnnealingWarmRestarts (T_0=10, T_mult=2) |
| 精度 | AMP 混合精度 + 梯度裁剪 (max_norm=1.0) |
| 早停 | 连续 50 轮 Val F1 无提升 |

## 致谢与第三方依赖

本项目的**仿真数据集生成**依赖以下开源项目，仅作为数据生成工具使用，其代码**不属于本项目作品，也未收录进本仓库**：

- **SeaClutterSuppression** — 作者 Pepijn Lens（Leiden University BSc thesis / TNO 实习）
  <https://github.com/pepijn-lens/SeaClutterSuppression>
  用于生成雷达 RDM 仿真数据与对应的目标掩膜。使用请遵循原项目 LICENSE。

本仓库的 `functions.py`（RTFEN 网络本体）、训练脚本与评估逻辑为本人实现。
