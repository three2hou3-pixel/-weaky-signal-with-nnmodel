import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================== 时序特征增强模块 ======================
# 1. 时序压缩模块
class TemporalCompressionModule(nn.Module):
    def __init__(self, in_channels=1, L=32):
        super(TemporalCompressionModule, self).__init__()
        # 第一级3D卷积层：通道数从 1 扩展到 L(32)
        self.conv_stage1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=L, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(L),
            nn.ReLU(inplace=True)
        )
        # 第二级3D卷积层：通道数从 L(32) 扩展到 2L(64)
        self.conv_stage2 = nn.Sequential(
            nn.Conv3d(in_channels=L, out_channels=2*L, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(2*L),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 输入 x 的形状为 (B, 1, T, H, W)
        x = self.conv_stage1(x)  # 输出形状为 (B, L, T, H, W)
        x = self.conv_stage2(x)  # 输出形状为 (B, 2L, T, H, W)
        return x
    
if __name__ == "__main__":
    # 模拟输入参数：Batch=2, Channels=1, 序列长度T=50, 距离H=128, 多普勒W=128
    B, C, T, H, W = 2, 1, 50, 128, 128
    input_tensor = torch.randn(B, C, T, H, W)  # 模拟输入数据
    model = TemporalCompressionModule(in_channels=1, L=32)  # 创建模型实例
    output = model(input_tensor)  # 前向传播
    print("输入数据维度:", input_tensor.shape)  # 输出输入张量的形状
    print("输出数据维度:", output.shape)  # 输出输出张量的形状

# 2. 双向ConvLSTM模块(Bi-ConvLSTM)
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size[0] // 2       # 保持输入输出尺寸不变
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                            out_channels=4 * hidden_dim,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=bias)
        
    def forward(self, x, cur_state):
        """
        x: 输入张量 (B, C, H, W) 当前时刻t的输入
        cur_state: 包含上一时刻的(h_prev, c_prev)
        """
        h_prev, c_prev = cur_state
        combined = torch.cat([x, h_prev], dim=1)   # 在通道维度拼接输入和隐状态： [B, input_dim + hidden_dim, H, W]
        gates = self.conv(combined)
        # 分割卷积输出为四个部分：输入门、遗忘门、输出门和候选记忆
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        
        # 应用激活函数
        i = torch.sigmoid(i)    # 输入门
        f = torch.sigmoid(f)    # 遗忘门
        o = torch.sigmoid(o)    # 输出门
        g = torch.tanh(g)       # 候选记忆
        # 更新细胞状态和隐状态
        c_next = f * c_prev + i * g  # 更新细胞状态
        h_next = o * torch.tanh(c_next)  # 更新隐状态

        return h_next, c_next
    
class BiConvLSTM(nn.Module):
    '''
    input_dim: 2 * L 
    hidden_dim: 2 * L
    '''
    def __init__(self, input_dim, hidden_dim, kernel_size=(3,3)):
        super(BiConvLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # 前向和后向的ConvLSTM单元
        self.forward_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.backward_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
    
    def forward(self, x):
        """
        输入维度：[B, C, T, H, W] (这里的 C 对应论文中的 2L = 64)
        """
        B, C, T, H, W = x.size()
        # 初始化前向和后向的隐状态和细胞状态
        h_fw = torch.zeros(B, self.hidden_dim, H, W, device=x.device)   # 前向隐状态
        c_fw = torch.zeros(B, self.hidden_dim, H, W, device=x.device)   # 前向细胞状态

        h_bw = torch.zeros(B, self.hidden_dim, H, W, device=x.device)   # 后向隐状态
        c_bw = torch.zeros(B, self.hidden_dim, H, W, device=x.device)   # 后向细胞状态

        out_fw = []
        out_bw = []

        # 1. 前向时间循环(t = 0 到 T-1)
        for t in range(T):
            h_fw, c_fw = self.forward_cell(x[:, :, t, :, :],(h_fw, c_fw)) # 前向处理当前时间步
            out_fw.append(h_fw)  # 保存前向隐状态输出
        # 2. 后向时间循环(t = T-1 到 0)
        for t in reversed(range(T)):
            h_bw, c_bw = self.backward_cell(x[:, :, t, :, :],(h_bw, c_bw)) # 后向处理当前时间步
            out_bw.append(h_bw)  # 保存后向隐状态输出
        
        # 调整后向列表顺序，使其时间轴与前向对齐
        out_bw = out_bw[::-1]  # 反转后向输出列表

        # 将列表堆叠为张量：[B, hidden_dim, T, H, W]
        out_fw = torch.stack(out_fw, dim=2)  # 前向输出
        out_bw = torch.stack(out_bw, dim=2)  # 后向输出

        output = torch.cat([out_fw, out_bw], dim=1)  # 在通道维度拼接前向和后向输出：[B, 2*hidden_dim, T, H, W]
        return output

# 3. 时间注意力机制与多尺度融合模块
class TemporalAttentionFusion(nn.Module):
    def __init__(self, L=32, T=50):
        """
        L: 基础通道数(论文为默认32)
        T: 序列长度(论文为默认50)
        """
        super(TemporalAttentionFusion, self).__init__()
        self.L = L
        self.T = T
        in_channels = 4 * L     # Bi-ConvLSTM 输出通道数 (4L = 128)

        # 1. 时间注意力机制
        self.attention_net = nn.Sequential(
            # 第一层：3x3卷积，输入通道为4L，输出通道为2L
            nn.Conv2d(in_channels=in_channels * T, out_channels=2*L, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 第二层：3x3卷积，输入通道为2L，输出通道L
            nn.Conv2d(in_channels=2*L, out_channels=2*L, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 第三层：1x1卷积，输入通道为L，输出通道为T，并用Sigmoid激活函数生成注意力权重
            nn.Conv2d(in_channels=2*L, out_channels=T, kernel_size=1,padding=0),
            nn.Sigmoid()
        )

        # 2. 多尺度融合模块(将加权后的5D张量融合为4D张量给U-Net)
        self.fusion_net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * T, out_channels=in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, F):
        """
        F: Bi-ConvLSTM 输出的特征图，维度为 [B, 4L, T, H, W]
        """
        B, C, T, H, W = F.size()  # C 应该等于 4L (128)

        # 1. 特征维度转换
        F_prime = F.permute(0, 2, 1, 3, 4).reshape(B, T * C, H, W)  # 转换为 [B, C*T, H, W]
        
        # 2. 计算时间注意力权重
        A = self.attention_net(F_prime)  # 输出维度为 [B, T, H, W]

        # 3. 特征加权
        A_expanded = A.unsqueeze(1)
        F_weighted = F * A_expanded  # 广播机制加权，结果维度仍为 [B, 4L, T, H, W]

        # 4. 多尺度融合
        F_weighted_flat = F_weighted.permute(0, 2, 1, 3, 4).reshape(B, T * C, H, W)  # 转换为 [B, C*T, H, W]

        # 通过融合网络得到最终的融合特征图，消除了T维，输出 U-Net 能够接受的 [B, 4L, H, W]
        out_fused = self.fusion_net(F_weighted_flat)
        return out_fused, A     # 返回融合后的特征图和注意力权重
    

# ================= 通信与维度验证 =================
if __name__ == "__main__":
    # 模拟 Bi-ConvLSTM 输出参数: Batch=2, 通道4L=128, T=50帧, H=128, W=128
    B, L, T, H, W = 2, 32, 50, 128, 128
    dummy_bilstm_out = torch.randn(B, 4 * L, T, H, W)
    
    # 实例化注意力与融合模块
    attention_module = TemporalAttentionFusion(L=L, T=T)
    
    output_fused, attention_map = attention_module(dummy_bilstm_out)
    
    print(f"Bi-ConvLSTM 输入维度: {dummy_bilstm_out.shape}")
    print(f"生成注意力图(A)维度:  {attention_map.shape}")
    print(f"输入给U-Net的最终维度: {output_fused.shape}")



# ====================== U-Net 模块 ======================
class DoubleConv(nn.Module):
    """
    U-Net 的基础模块:连续两次的3x3卷积,每次卷积后接BatchNorm和ReLU激活函数
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            # 第一次卷积，不改变尺寸
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 第二次卷积，不改变尺寸
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),    
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)
    
class UNet(nn.Module):
    def __init__(self, L=32):
        """
        基于雷达时序增强网络的 U-Net 主干
        输入维度： [B, 4L, H, W] (来自 TemporalAttentionFusion 模块的输出)
        输出维度： [B, 4L, H, W] (二分类分割掩码) 
        """
        super(UNet, self).__init__()

        in_channels = 4 * L # 输入通道数 (128)

        # ============ 编码器部分 ============
        # 第 1 级：双卷积 + 下采样 ([3x3, 4L] x 2)
        self.enc1 = DoubleConv(in_channels, 4 * L)   # 输入4L，输出4L
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第 2 级：双卷积 + 下采样 ([3x3, 8L] x 2)
        self.enc2 = DoubleConv(4 * L, 8 * L)        # 输入4L，输出8L
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ============ 瓶颈层 ============
        self.bottleneck = DoubleConv(8 * L, 16 * L)  # 输入8L，输出16L

        # ============ 解码器部分 ============
        # 第 1 级上采样：恢复到 1/2 分辨率
        self.upconv1 = nn.ConvTranspose2d(16 * L, 8 * L, kernel_size=2, stride=2)  # 上采样，输入16L，输出8L
        # 上采样后与 enc2 进行跳跃链接(Concat),通道数变为 8L + 8L = 16L
        self.dec1 = DoubleConv(16 * L, 8 * L)  # 输入16L，输出8L
        # 第 2 级上采样：恢复到原始分辨率
        self.upconv2 = nn.ConvTranspose2d(8 * L, 4 * L,kernel_size=2, stride=2)
        self.dec2 = DoubleConv(8 * L, 4 * L)

    def forward(self, x):
        # --- 编码路径 ---
        enc1_out = self.enc1(x)  # 编码器第一层输出，维度 [B, 4L, H, W]
        enc1_pooled = self.pool1(enc1_out)  # 池化后维度 [B, 4L, H/2, W/2]

        enc2_out = self.enc2(enc1_pooled)  # 编码器第二层输出，维度 [B, 8L, H/2, W/2]
        enc2_pooled = self.pool2(enc2_out)  # 池化后维度 [B, 8L, H/4, W/4]
        
        # --- 瓶颈层 ---
        bottleneck_out = self.bottleneck(enc2_pooled)  # 瓶颈层输出，维度 [B, 16L, H/4, W/4]
        
        # --- 解码路径 ---
        up1 = self.upconv1(bottleneck_out)  # 上采样，维度 [B, 8L, H/2, W/2]
        concat1 = torch.cat([up1, enc2_out], dim=1)  # 跳跃连接: 与 enc2_out 拼接，维度 [B, 16L, H/2, W/2]
        dec1_out = self.dec1(concat1)  # 解码器第一层输出，维度 [B, 8L, H/2, W/2]

        up2 = self.upconv2(dec1_out)  # 上采样，维度 [B, 4L, H, W]
        concat2 = torch.cat([up2, enc1_out], dim=1)  # 跳跃连接: 与 enc1_out 拼接，维度 [B, 8L, H, W]
        dec2_out = self.dec2(concat2)  # 解码器第二层输出，维度 [B, 4L, H, W]
        
        return dec2_out
    

# ================= 通信与维度验证 =================
if __name__ == "__main__":
    # 模拟 TemporalAttentionFusion 输出的融合特征数据
    B, L, H, W = 2, 32, 128, 128
    fusion_output = torch.randn(B, 4 * L, H, W)
    # 实例化 U-Net 模块
    unet = UNet(L=L)

    # 打印网络参数量
    total_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"U-Net 模块的总参数量: {total_params:,} 个")

    unet_output = unet(fusion_output)
    print(f"输入 U-Net 的维度: {fusion_output.shape}")
    print(f"U-Net 输出的维度: {unet_output.shape}")



# ====================== 多帧预测头模块 ======================
# 多帧预测头网络
class MultiFramePredictionHead(nn.Module):
    def __init__(self, L=32, T=50):
        """
        多帧预测头网络：
        参数：
        L: 基础通道数(论文为默认32)
        T: 预测帧数(论文为默认50)
        """
        super(MultiFramePredictionHead, self).__init__()
        self.L = L
        self.T = T

        # =========== 共享特征提取 ===========
        # U-Net 输出的特征图维度为 [B, 4L, H, W]
        # 两层卷积：[3x3, 2L] 和 [3x3, L]
        self.shared_extractor = nn.Sequential(
            nn.Conv2d(in_channels=4 * L, out_channels=2 * L, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * L),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=2 * L,out_channels=L, kernel_size=3, padding=1),
            nn.BatchNorm2d(L),
            nn.ReLU(inplace=True)
        )

        # =========== 并行预测头 ===========
        # 对应论文表1： [[1x1, 1] x T]，每个头独立预测一帧
        # 使用 nn.ModuleList 来包装 T 个完全独立、参数不共享的 1x1 卷积层
        self.parallel_heads = nn.ModuleList([
            nn.Conv2d(in_channels=L, out_channels=1, kernel_size=1)
            for _ in range(T)
        ])

    def forward(self, x, original_size=(128, 128)):
        """
        x: U-Net 输出的特征图，维度为 [B, 4L, H, W]
        original_size: 原始输入雷达图的空间分辨率
        """

        # 1. 共享特征提取
        shared_features = self.shared_extractor(x)  # 输出维度为 [B, L, H, W]

        # 2. 并行预测头
        outputs = []
        for t in range(self.T):
            head_out = self.parallel_heads[t](shared_features)

            # 双线性插值采样
            head_out = F.interpolate(head_out, size=original_size, mode='bilinear', align_corners=False)
            # 使用 Sigmoid 激活，将像素值转化为 [0, 1] 之间的概率值，越接近1表示越可能是目标
            prob_map = torch.sigmoid(head_out)
            outputs.append(prob_map)  # 每个元素维度为 [B, 1, H, W]
        
        # 将生成的 T 张独立概率图再时间维度(dim=1)进行堆叠，最终输出维度：[B, T, 1, H, W]
        final_output = torch.stack(outputs, dim=1)

        return final_output

# ================= 通信与维度验证 =================
if __name__ == "__main__":
    B, L, T, H, W = 2, 32, 50, 128, 128
    
    unet_output = torch.randn(B, 4 * L, H, W)  # 模拟 U-Net 输出的特征图
    # 实例化多帧预测头网络
    prediction_head = MultiFramePredictionHead(L=L, T=T)

    final_masks = prediction_head(unet_output, original_size=(H, W))
    print(f"输入 MultiFramePredictionHead 的维度: {unet_output.shape}")
    print(f"输出 MultiFramePredictionHead 的维度: {final_masks.shape}")



# ====================== 时序损失函数 ======================
# 时序损失函数(去掉权重衰减系数)
class TemporalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.8, smooth=1e-6):
        """
        基于时序衰减的 Tversky Loss
        参数：
        alpha: FP 的惩罚权重 (论文中为 0.2)
        beta: FN 的惩罚权重 (论文中为 0.8)
        gamma: 时间衰减因子 (0, 1)
        smooth: 平滑因子 (避免除零错误)
        """
        super(TemporalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, y_pred, y_true):
        """
        y_pred: 多帧预测头输出，维度为 [B, T, 1, H, W](经过了 Sigmoid)
        y_true: 真实的掩膜标签，维度为 [B, T, 1, H, W] (值为 0 或 1)
        """

        # 1. 展平空间维度：把 [B, T, 1, H, W] 变成 [B, T, H*W]
        B, T, C, H, W = y_pred.size()
        pred_flat = y_pred.view(B, T, -1)
        true_flat = y_true.view(B, T, -1)

        # 结果的维度都是 [B, T]
        TP = (pred_flat * true_flat).sum(dim=2)               # 真正例 (预测对的目标)
        FP = (pred_flat * (1 - true_flat)).sum(dim=2)         # 假正例 (把杂波当目标)
        FN = ((1 - pred_flat) * true_flat).sum(dim=2)         # 假阴性 (漏掉的真实目标)

        # 3. 计算每帧的 Tversky Index
        # 维度 [B, T]
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        # 4. 转化为 Loss (1 - Index)
        frame_losses = 1.0 - tversky_index  # 维度 [B, T]

        # 5. 直接对所有帧、所有 Batch 求平均
        # 等价于给这 T 帧赋予了完全相同的权重 (1/T)
        final_loss = frame_losses.mean()

        return final_loss
    
# ================= 损失函数测试 =================
if __name__ == "__main__":
    B, T, C, H, W = 2, 50, 1, 128, 128
    # 模拟多帧预测头的输出 (经过 Sigmoid 激活，值在 [0, 1] 之间)
    y_pred = torch.rand(B, T, C, H, W)
    # 模拟真实标签 (二分类掩膜，值为 0 或 1)
    y_true = torch.randint(0, 2, (B, T, C, H, W)).float()

    # 实例化损失函数
    criterion = TemporalTverskyLoss(alpha=0.2, beta=0.8)
    loss = criterion(y_pred, y_true)
    print(f"Temporal Tversky Loss: {loss.item()}")



# ====================== RFTEN模型集成 ======================
class RTFEN(nn.Module):
    '''
    in_channel: 原始输入通道
    L: 基础通道数(论文中设为32)
    T: 序列长度(论文中设为50)
    original_size: 图像大小
    '''
    def __init__(self, in_channels=1, L=32, T=50, original_size=(128,128)):
        super(RTFEN, self).__init__()
        self.original_size = original_size
        # 1. 时序压缩模块
        self.temporal_compression = TemporalCompressionModule(in_channels=in_channels, L=L)  
        # 2. 双向ConvLSTM时序建模模块
        self.biconvlstm = BiConvLSTM(input_dim=2*L, hidden_dim=2*L)         
        # 3. 注意力机制和多尺度融合模块
        self.attention_fusion = TemporalAttentionFusion(L=L, T=T)             
        # 4. U-Net
        self.unet = UNet(L=L)
        # 5. 多帧预测头                                       
        self.multi_head = MultiFramePredictionHead(L=L, T=T)              
        
    def forward(self, x):
        """
        x: 初始雷达时序特征图，维度 [B, 1, T, H, W]
        """
        # 1. 提取初级空时特征 -> [B, 2L, T, H, W]
        feat_compressed = self.temporal_compression(x)
        
        # 2. 双向时序演化建模 -> [B, 4L, T, H, W]
        feat_lstm = self.biconvlstm(feat_compressed)
        
        # 3. 计算注意力并压缩时间维度
        # 注意这里：显式地将元组拆包，丢弃/保存注意力图 A，只把融合特征传给下游
        feat_fused, attention_map = self.attention_fusion(feat_lstm) # feat_fused: [B, 4L, H, W]
        
        # 4. U-Net 提取深层空间特征 -> [B, 4L, H, W]
        feat_unet = self.unet(feat_fused)
        
        # 5. 多帧并行预测并上采样 -> [B, T, 1, H, W]
        final_masks = self.multi_head(feat_unet, original_size=self.original_size)
        
        # 如果你未来想在 TensorBoard 里可视化注意力图，可以将 attention_map 也 return 出来
        # 输出预测结果，维度为 [B, T, 1, H, W]
        return final_masks