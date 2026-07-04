import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========================================================================
# 0. 基础组件与 VMamba 导入 (带 Dummy Fallback)
# =========================================================================
try:
    from submodules.vmamba import VSSM, VSSBlock, LayerNorm, Permute
except ImportError:
    try:
        from models.submodules.vmamba import VSSM, VSSBlock, LayerNorm, Permute
    except ImportError:
        print("【提示】未找到 vmamba.py，使用内置 Dummy 模块演示结构运行。")

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# =========================================================================
# 1. 置信度感知几何膨胀模块 (Geometric Prior)
# =========================================================================
class ConfidenceAwareGeometricDilation(nn.Module):
    """
    结合传统几何形态学(膨胀)与深度学习(置信度预测)的先验模块。
    输出:
      1. Weighted Depth: 原始点保留 + 膨胀点加权
      2. Confidence Map: 原始点为1 + 膨胀点预测值
    """

    def __init__(self, channels=1, expansion_steps=3):
        super().__init__()
        self.expansion_steps = expansion_steps

        # 不可学习: 最大池化模拟膨胀
        self.dilation_op = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

        # 可学习: 轻量级置信度预测网络
        self.conf_net = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_lidar):
        # x_lidar: (B, 1, H, W)

        # 1. 几何膨胀 (No Grad)
        with torch.no_grad():
            x_coarse = x_lidar
            for _ in range(self.expansion_steps):
                x_coarse = self.dilation_op(x_coarse)

            mask_orig = (x_lidar > 0).float()
            mask_coarse = (x_coarse > 0).float()

        # 2. 预测置信度
        raw_confidence = self.conf_net(x_coarse)

        # 3. 融合逻辑
        # 原始点置信度强制为 1.0
        final_confidence = mask_orig * 1.0 + (1 - mask_orig) * raw_confidence
        # 空洞处强制为 0
        final_confidence = final_confidence * mask_coarse

        # 4. 生成加权几何先验图
        x_weighted = x_coarse * final_confidence

        return x_weighted, final_confidence


# =========================================================================
# 2. 下采样与上采样模块
# =========================================================================
class Downsample(nn.Module):
    """ NHWC -> NCHW Conv -> NHWC """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1)
        )

    def forward(self, x):
        return self.net(x)


class PatchExpand(nn.Module):
    """ NHWC -> Linear -> PixelShuffle -> NHWC """

    def __init__(self, dim, scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, (dim // 2) * (scale ** 2), bias=False)
        self.norm = norm_layer(dim // 2)
        self.scale = scale

    def forward(self, x):
        x = self.expand(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.pixel_shuffle(x, self.scale)
        x = x.permute(0, 2, 3, 1).contiguous()
        return self.norm(x)


# =========================================================================
# 3. SAMA & SADC (核心交互模块)
# =========================================================================
class SparseMetricAnchoring(nn.Module):
    """ LiDAR -> Event: 稀疏锚定 """

    def __init__(self, dim):
        super().__init__()
        self.event_proj = nn.Linear(dim, dim)
        self.lidar_proj = nn.Linear(dim, dim)
        self.validity_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        self.correction_layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_event, x_lidar):
        anchor_conf = self.validity_gate(x_lidar)
        metric_gap = self.lidar_proj(x_lidar) - self.event_proj(x_event)
        correction = self.correction_layer(metric_gap)
        x_event_anchored = x_event + anchor_conf * correction
        return self.norm(x_event_anchored)


class StructureAwareDiffusionController(nn.Module):
    """ Event -> LiDAR: 结构扩散控制 """

    def __init__(self, dim):
        super().__init__()
        self.gradient_sensor = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            LayerNorm2d(dim),
            nn.SiLU()
        )
        self.affinity_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_lidar, x_event):
        x_e_in = x_event.permute(0, 3, 1, 2).contiguous()
        local_structure = self.gradient_sensor(x_e_in)
        affinity = self.affinity_proj(local_structure)
        affinity = affinity.permute(0, 2, 3, 1).contiguous()
        return x_lidar * affinity


# =========================================================================
# 4. CMAM - 互补流形对齐模块
# =========================================================================
class CMAM(nn.Module):
    def __init__(self, dim, ssm_kwargs):
        super().__init__()
        self.sama = SparseMetricAnchoring(dim)
        self.event_mamba = VSSBlock(hidden_dim=dim, **ssm_kwargs)

        self.affinity_ctrl = StructureAwareDiffusionController(dim)
        self.lidar_mamba = VSSBlock(hidden_dim=dim, **ssm_kwargs)

        self.conf_pred = nn.Sequential(
            nn.Linear(dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        self.fusion_out = nn.Linear(dim * 2, dim)

    def forward(self, x_l, x_e):
        # 1. SAMA Update
        x_e_anchored = self.sama(x_e, x_l)
        x_e_new = self.event_mamba(x_e_anchored)

        # 2. Affinity Update
        x_l_gated = self.affinity_ctrl(x_l, x_e_new)
        x_l_new = self.lidar_mamba(x_l_gated)

        # 3. Fusion
        cat_feat = torch.cat([x_l_new, x_e_new], dim=-1)
        weights = self.conf_pred(cat_feat)
        w_l, w_e = weights.chunk(2, dim=-1)

        x_fused_base = w_l * x_l_new + w_e * x_e_new
        x_fused = self.fusion_out(torch.cat([x_fused_base, x_e_new], dim=-1))

        return x_l_new, x_e_new, x_fused


# =========================================================================
# 5. 主网络: LEDepth
# =========================================================================
class LEDepth(nn.Module):
    def __init__(
            self,
            lidar_chans=1, event_chans=4, out_chans=1,
            patch_size=4,
            depths=[2, 2, 18, 2],
            dims=96,
            decoder_depths=[2, 2, 2],
            ssm_d_state=16, ssm_ratio=2.0, ssm_init="v0", forward_type="v05_noz",
            imgsize=224,
            **kwargs,
    ):
        super().__init__()

        # 防守性编程：防止 patch_size 为 None
        if patch_size is None: patch_size = 4

        self.num_enc_layers = len(depths)
        self.total_factor = patch_size * (2 ** (self.num_enc_layers - 1))
        self.dims = [int(dims * 2 ** i) for i in range(self.num_enc_layers)] if isinstance(dims, int) else dims
        self.patch_size = patch_size

        ssm_kwargs = dict(
            ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, ssm_init=ssm_init, forward_type=forward_type,
            channel_first=False
        )

        # 1. 几何先验网络
        self.geom_prior_net = ConfidenceAwareGeometricDilation(channels=lidar_chans)

        # 2. Embedding
        # 输入通道 = Lidar(1) + Mask(1) + PriorDepth(1) + PriorConf(1) = 4
        self.lidar_patch_embed = VSSM._make_patch_embed(4, self.dims[0], patch_size, True, "v1")
        self.event_patch_embed = VSSM._make_patch_embed(event_chans, self.dims[0], patch_size, True, "v1")
        self.pos_embed = VSSM._pos_embed(self.dims[0], patch_size, imgsize)

        # 3. Encoder
        self.enc_stages = nn.ModuleList()
        self.downsamples_l = nn.ModuleList()
        self.downsamples_e = nn.ModuleList()

        for i in range(self.num_enc_layers):
            stage_blocks = nn.ModuleList([
                CMAM(self.dims[i], ssm_kwargs)
                for _ in range(depths[i])
            ])
            self.enc_stages.append(stage_blocks)

            if i < self.num_enc_layers - 1:
                self.downsamples_l.append(Downsample(self.dims[i], self.dims[i + 1]))
                self.downsamples_e.append(Downsample(self.dims[i], self.dims[i + 1]))

        # 4. Bottleneck
        self.bottleneck = CMAM(self.dims[-1], ssm_kwargs)
        self.bottleneck_reduce = nn.Linear(self.dims[-1], self.dims[-1])  # 保持接口一致

        # 5. Decoder (正确堆叠 Block)
        self.decoder_layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.reduce_channels = nn.ModuleList()

        for i in range(self.num_enc_layers - 2, -1, -1):
            self.upsamples.append(PatchExpand(dim=self.dims[i + 1]))
            self.reduce_channels.append(nn.Linear(self.dims[i] * 2, self.dims[i]))

            # 使用 Sequential 堆叠 decoder_depths[i] 个 Block
            blocks = [VSSBlock(hidden_dim=self.dims[i], **ssm_kwargs) for _ in range(decoder_depths[i])]
            self.decoder_layers.append(nn.Sequential(*blocks))

        # 6. Head
        self.head_up1 = nn.ConvTranspose2d(self.dims[0], self.dims[0] // 2, 2, 2)
        self.head_norm1 = LayerNorm2d(self.dims[0] // 2)
        self.head_up2 = nn.ConvTranspose2d(self.dims[0] // 2, 32, 2, 2)
        self.head_norm2 = LayerNorm2d(32)
        self.head_last = nn.Conv2d(32, out_chans, 3, 1, 1)

        self.apply(VSSM(depths=[2])._init_weights)
        self.saved_lidar = None

    def forward(self, x_lidar, x_event, verbose=False):
        # --- Preprocessing & Padding ---
        if x_lidar is None:
            if self.saved_lidar is not None:
                x_lidar = self.saved_lidar.clone()
            else:
                x_lidar = torch.zeros_like(x_event[:, :1])
        else:
            self.saved_lidar = x_lidar.clone()

        B, _, H, W = x_lidar.shape
        factor = self.total_factor
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        if pad_h > 0 or pad_w > 0:
            x_lidar = F.pad(x_lidar, (0, pad_w, 0, pad_h))
            x_event = F.pad(x_event, (0, pad_w, 0, pad_h))

        # --- A. 几何先验生成 (在 Embedding 之前) ---
        # 产生 prior_depth 和 prior_conf
        prior_depth, prior_conf = self.geom_prior_net(x_lidar)

        valid_mask = (x_lidar > 0).float()

        # 拼接 4 通道输入
        x_lidar_in = torch.cat([x_lidar, valid_mask, prior_depth, prior_conf], dim=1)

        # --- B. Embedding ---
        x_l = self.lidar_patch_embed(x_lidar_in)
        x_e = self.event_patch_embed(x_event)

        # 强制 NHWC
        if x_l.shape[1] == self.dims[0]:
            x_l = x_l.permute(0, 2, 3, 1).contiguous()
            x_e = x_e.permute(0, 2, 3, 1).contiguous()

        # --- C. Pos Embed ---
        if self.pos_embed is not None:
            pos = self.pos_embed
            curr_h, curr_w = x_l.shape[1], x_l.shape[2]

            # 统一转 NCHW 插值
            if pos.shape[1] != self.dims[0] and pos.shape[-1] == self.dims[0]:
                pos = pos.permute(0, 3, 1, 2)
            if pos.shape[2] != curr_h or pos.shape[3] != curr_w:
                pos = F.interpolate(pos, size=(curr_h, curr_w), mode='bilinear', align_corners=False)

            # 转回 NHWC 并对齐 Batch
            pos = pos.permute(0, 2, 3, 1).contiguous()
            if pos.shape[0] != B:
                pos = pos.repeat(B, 1, 1, 1)

            x_l = x_l + pos
            x_e = x_e + pos

        skips = []

        # --- D. Encoder Loop ---
        for i, blocks in enumerate(self.enc_stages):
            if verbose: print(f"Stage {i} Input: {x_l.shape}")
            for blk in blocks:
                x_l, x_e, x_fused = blk(x_l, x_e)

            skips.append(x_fused)

            if i < self.num_enc_layers - 1:
                x_l = self.downsamples_l[i](x_l)
                x_e = self.downsamples_e[i](x_e)

        # --- E. Bottleneck ---
        _, _, x_fused = self.bottleneck(x_l, x_e)
        if verbose: print(f"Bottleneck Out: {x_fused.shape}")

        # --- F. Decoder Loop ---
        for i, (up, reduce, layer) in enumerate(zip(self.upsamples, self.reduce_channels, self.decoder_layers)):
            idx = self.num_enc_layers - 2 - i

            x_up = up(x_fused)
            skip = skips[idx]

            if verbose: print(f"Decoder {i}: Up={x_up.shape}, Skip={skip.shape}")

            x_cat = torch.cat([x_up, skip], dim=-1)
            x_fused = reduce(x_cat)
            x_fused = layer(x_fused)

        # --- G. Head (NCHW) ---
        # 1. NHWC -> NCHW
        x_fused = x_fused.permute(0, 3, 1, 2).contiguous()

        # 2. Upsample + Norm (LayerNorm2d 处理 NCHW)
        out = self.head_up1(x_fused)
        out = self.head_norm1(out)
        out = F.gelu(out)

        out = self.head_up2(out)
        out = self.head_norm2(out)
        out = F.gelu(out)

        out = self.head_last(out)

        # Unpadding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out


# =========================================================================
# 测试脚本 (Main)
# =========================================================================
import time

# =========================================================================
# 测试脚本 (Main)
# =========================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Running Final LEDepth on {device}...")

    # 配置模型参数
    model = LEDepth(
        lidar_chans=1,
        event_chans=4,
        out_chans=1,
        patch_size=4,
        dims=96,
        depths=[2, 2, 18, 2],
        decoder_depths=[2, 2, 2]
    ).to(device)

    # 模拟数据 (720x1280 不规则分辨率测试)
    # h, w = 720, 1280
    h, w = 720, 1280
    x_lidar = torch.randn(1, 1, h, w).to(device)
    x_event = torch.randn(1, 4, h, w).to(device)

    print(f"📦 Input: LiDAR {x_lidar.shape}, Event {x_event.shape}")

    # ==========================================
    # 1. 计算参数量 (Parameters)
    # ==========================================
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数量 (Trainable Params): {total_params / 1e6:.2f} M")

    # ==========================================
    # 2. 计算计算量 (FLOPs / MACs)
    # ==========================================
    try:
        from thop import profile
        # thop 通常计算的是 MACs (乘加操作数)，1 MAC ≈ 2 FLOPs
        print("🧮 正在计算 FLOPs (可能会花费几秒钟)...")
        macs, params = profile(model, inputs=(x_lidar, x_event), verbose=False)
        print(f"🧮 计算量 (MACs): {macs / 1e9:.2f} G")
        print(f"🧮 预估 FLOPs: {(macs * 2) / 1e9:.2f} G")
    except ImportError:
        print("【提示】未检测到 thop 库，跳过 FLOPs 计算。建议运行 `pip install thop` 后重试。")

    # ==========================================
    # 3. 计算单次运行时间 (Inference Time)
    # ==========================================
    try:
        print("⏱️ 开始测速 (包含 Warm-up)...")
        model.eval() # 切换到评估模式
        with torch.no_grad():
            # 预热 (Warm-up): 避免 GPU 冷启动导致的首次运行耗时过长虚高
            for _ in range(5):
                _ = model(x_lidar, x_event, verbose=False)

            # 严格同步 GPU 开始时间
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # 实际测速运行
            out = model(x_lidar, x_event, verbose=False)

            # 严格同步 GPU 结束时间
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

        infer_time_ms = (end_time - start_time) * 1000
        print(f"⚡ 单次前向传播时间: {infer_time_ms:.2f} ms (约 {1000/infer_time_ms:.1f} FPS)")

        # 验证输出
        print(f"✅ Success! Output Shape: {out.shape}")
        if out.shape[-2:] == (h, w):
            print("🎉 Resolution Match!")
        else:
            print(f"⚠️ Resolution Mismatch: Expected {(h, w)}, Got {out.shape[-2:]}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    def measure_memory_cost(model, dummy_inputs):
        # 1. 清空之前的显存缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 2. 记录基础显存（模型权重 + CUDA上下文）
        base_memory = torch.cuda.memory_allocated() / (1024 ** 3)

        # 3. 运行一次前向传播
        model.eval()
        with torch.no_grad():
            _ = model(*dummy_inputs)

        # 4. 获取峰值显存 (Peak Memory)
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

        print(f"📦 基础显存 (模型权重): {base_memory:.3f} GB")
        print(f"📈 峰值显存 (GPU Memory Cost): {peak_memory:.3f} GB")

    # 使用示例 (假设使用你之前的 ALED 模型变量):
    measure_memory_cost(model, (x_lidar, x_event))

if __name__ == "__main__":
    main()
