import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========================================================================
# 0. 基础组件与 VMamba 导入
# =========================================================================
try:
    from submodules.vmamba import VSSM, VSSBlock, LayerNorm, Permute
except ImportError:
    try:
        from models.submodules.vmamba import VSSM, VSSBlock, LayerNorm, Permute
    except ImportError:
        print("【警告】找不到 vmamba.py，将使用 DummyBlock 代替进行结构演示。")


        class Permute(nn.Module):
            def __init__(self, *args): super().__init__(); self.args = args

            def forward(self, x): return x.permute(*self.args)


        class LayerNorm(nn.Module):
            def __init__(self, dim, eps=1e-6): super().__init__(); self.norm = nn.LayerNorm(dim, eps=eps)

            def forward(self, x): return self.norm(x)


        class VSSBlock(nn.Module):
            def __init__(self, hidden_dim, **kwargs): super().__init__(); self.net = nn.Linear(hidden_dim, hidden_dim)

            def forward(self, x): return self.net(x) + x  # Residual


        class VSSM:
            @staticmethod
            def _make_patch_embed(in_chans, dim, patch_size, norm, version):
                return nn.Sequential(
                    nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size),
                    Permute(0, 2, 3, 1)
                )

            @staticmethod
            def _pos_embed(dim, patch_size, img_size):
                return torch.zeros(1, dim, img_size // patch_size, img_size // patch_size)

            @staticmethod
            def _make_downsample(dim, out_dim, version):
                return nn.Identity()


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


# =========================================================================
# 1. 下采样模块 (NHWC)
# =========================================================================
class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================================
# 2. SAMA - 稀疏感知度量锚定 [LiDAR -> Event]
# =========================================================================
class SparseMetricAnchoring(nn.Module):
    """
        [LiDAR -> Event 方向]
        物理意义：稀疏残差校正 (Sparse Residual Rectification)。
        Event 提供了稠密的相对特征，LiDAR 提供了稀疏的绝对特征。
        此模块计算两者在语义空间的偏差 (Metric Gap)，并利用 LiDAR 的有效性
        作为门控 (Validity Gate)，只在"有钉子"的地方修正 Event 特征。
        """
    def __init__(self, dim):
        super().__init__()
        self.event_proj = nn.Linear(dim, dim)
        self.lidar_proj = nn.Linear(dim, dim)
        # 2. 锚点置信度生成器 (Validity Gate)
        # 输入 LiDAR 特征，判断当前像素是否具备"锚定资格"
        # LiDAR 有值处 -> 1 (强行修正)，空洞处 -> 0 (保留 Event 原样)
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
        # A. 计算锚点置信度 (Anchor Confidence)
        anchor_conf = self.validity_gate(x_lidar)
        # B. 计算度量偏差 (Metric Gap)
        # Gap = 真值语义(LiDAR) - 估计语义(Event)
        metric_gap = self.lidar_proj(x_lidar) - self.event_proj(x_event)
        # C. 生成修正量
        correction = self.correction_layer(metric_gap)
        # D. 稀疏锚定 (Sparse Update)
        # F_new = F_old + Confidence * Correction
        # 在空洞处，anchor_conf 趋近 0，x_event 保持原样，避免被 0 值污染
        x_event_anchored = x_event + anchor_conf * correction
        return self.norm(x_event_anchored)


# =========================================================================
# 3. SADC - 结构感知扩散控制器 [Event -> LiDAR]
# =========================================================================
class StructureAwareDiffusionController(nn.Module):
    """
        [Event -> LiDAR 方向] 升级版
        物理意义：结构感知各向异性扩散 (Structure-Aware Anisotropic Diffusion)。
        不再是简单的逐像素门控，而是利用 3x3 局部感受野提取 Event 的梯度结构 (Gradient Structure)，
        以此作为物理屏障来约束 LiDAR 的扩散。
        """
    def __init__(self, dim):
        super().__init__()
        # 1. 结构提取器 (Structure Extractor)
        # 使用 DWConv (3x3) 模拟 Sobel/Laplacian 算子，提取局部边缘/纹理梯度
        # groups=dim 保证了通道独立性，参数量极小
        self.gradient_sensor = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            LayerNorm2d(dim),
            nn.SiLU()
        )
        # 2. 通道注意力 (Channel Selector)
        # Event 有很多通道，不是所有通道都包含几何边缘。
        # 我们需要筛选出那些"对结构敏感"的通道来生成亲和力。
        # 3. 亲和力映射 (Affinity Mapper)
        # 将提取到的结构信息映射为扩散系数 (0~1)
        # 逻辑：结构特征(梯度)越强 -> 输出越趋近于 0 (阻断)
        # 结构特征越弱(平坦) -> 输出越趋近于 1 (扩散)
        self.affinity_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_lidar, x_event):
        x_e_in = x_event.permute(0, 3, 1, 2).contiguous()  # 加上 contiguous 避免 DDP 警告
        # 这一步感知了 3x3 邻域，判断当前像素是否处于边缘带上
        local_structure = self.gradient_sensor(x_e_in)
        # 生成亲和力图 ---
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
        x_e_anchored = self.sama(x_e, x_l)
        x_e_new = self.event_mamba(x_e_anchored)
        x_l_gated = self.affinity_ctrl(x_l, x_e_new)
        x_l_new = self.lidar_mamba(x_l_gated)
        cat_feat = torch.cat([x_l_new, x_e_new], dim=-1)
        weights = self.conf_pred(cat_feat)
        w_l, w_e = weights.chunk(2, dim=-1)
        x_fused_base = w_l * x_l_new + w_e * x_e_new
        x_fused = self.fusion_out(torch.cat([x_fused_base, x_e_new], dim=-1))
        return x_l_new, x_e_new, x_fused


# =========================================================================
# 5. 上采样模块
# =========================================================================
class PatchExpand(nn.Module):
    def __init__(self, dim, scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, (dim // 2) * (scale ** 2), bias=False)
        self.norm = norm_layer(dim // 2)
        self.scale = scale

    def forward(self, x):
        x = self.expand(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.scale)
        x = x.permute(0, 2, 3, 1)
        return self.norm(x)


# =========================================================================
# 6. 主网络: FusionMamba (LEDepth) - 修复 Decoder
# =========================================================================
class LEDepth(nn.Module):
    def __init__(
            self,
            lidar_chans=1, event_chans=4, out_chans=1,
            patch_size=4,
            depths=[2, 2, 18, 2],
            dims=128,
            decoder_depths=[2, 2, 2],  # 这里的参数会被正确使用
            ssm_d_state=16, ssm_ratio=2.0, ssm_init="v0", forward_type="v05_noz",
            imgsize=224,
            **kwargs,
    ):
        super().__init__()

        self.num_enc_layers = len(depths)
        self.total_factor = patch_size * (2 ** (self.num_enc_layers - 1))
        self.dims = [int(dims * 2 ** i) for i in range(self.num_enc_layers)] if isinstance(dims, int) else dims
        self.patch_size = patch_size

        ssm_kwargs = dict(
            ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, ssm_init=ssm_init, forward_type=forward_type,
            channel_first=False
        )

        # 1. Embedding
        self.lidar_patch_embed = VSSM._make_patch_embed(lidar_chans + 1, self.dims[0], patch_size, True, "v1")
        self.event_patch_embed = VSSM._make_patch_embed(event_chans, self.dims[0], patch_size, True, "v1")
        self.pos_embed = VSSM._pos_embed(self.dims[0], patch_size, imgsize)

        # 2. Encoder
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

        # 3. Bottleneck
        self.bottleneck = CMAM(self.dims[-1], ssm_kwargs)
        self.bottleneck_reduce = nn.Linear(self.dims[-1], self.dims[-1])

        # 4. Decoder [修复：正确使用 decoder_depths]
        self.decoder_layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.reduce_channels = nn.ModuleList()

        for i in range(self.num_enc_layers - 2, -1, -1):
            self.upsamples.append(PatchExpand(dim=self.dims[i + 1]))
            self.reduce_channels.append(nn.Linear(self.dims[i] * 2, self.dims[i]))

            # [Fix] 创建多个 Block 并封装进 Sequential
            # 这样 layer(x) 调用时就会连续通过 [decoder_depths[i]] 个 Block
            blocks = [
                VSSBlock(hidden_dim=self.dims[i], **ssm_kwargs)
                for _ in range(decoder_depths[i])
            ]
            self.decoder_layers.append(nn.Sequential(*blocks))

        # 5. Head
        self.head_up1 = nn.ConvTranspose2d(self.dims[0], self.dims[0] // 2, 2, 2)
        self.head_norm1 = LayerNorm2d(self.dims[0] // 2)
        self.head_up2 = nn.ConvTranspose2d(self.dims[0] // 2, 32, 2, 2)
        self.head_norm2 = LayerNorm2d(32)
        self.head_last = nn.Conv2d(32, out_chans, 3, 1, 1)

        self.apply(VSSM(depths=[2])._init_weights)
        self.saved_lidar = None

    def forward(self, x_lidar, x_event, verbose=False):
        # ... Preprocessing ...
        if x_lidar is None:
            if self.saved_lidar is not None:
                x_lidar = self.saved_lidar.clone()
            else:
                x_lidar = torch.zeros_like(x_event[:, :1])
        else:
            self.saved_lidar = x_lidar.clone()

        valid_mask = (x_lidar > 0).float()
        x_lidar_in = torch.cat([x_lidar, valid_mask], dim=1)

        B, _, H, W = x_lidar.shape
        factor = self.total_factor
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        if pad_h > 0 or pad_w > 0:
            x_lidar_in = F.pad(x_lidar_in, (0, pad_w, 0, pad_h))
            x_event = F.pad(x_event, (0, pad_w, 0, pad_h))

        # ... Embedding ...
        x_l = self.lidar_patch_embed(x_lidar_in)
        x_e = self.event_patch_embed(x_event)

        if x_l.shape[1] == self.dims[0]:
            x_l = x_l.permute(0, 2, 3, 1)
            x_e = x_e.permute(0, 2, 3, 1)

        # ... Pos Embed ...
        if self.pos_embed is not None:
            pos = self.pos_embed
            curr_h, curr_w = x_l.shape[1], x_l.shape[2]
            if pos.shape[1] != self.dims[0] and pos.shape[-1] == self.dims[0]:
                pos = pos.permute(0, 3, 1, 2)
            if pos.shape[2] != curr_h or pos.shape[3] != curr_w:
                pos = F.interpolate(pos, size=(curr_h, curr_w), mode='bilinear', align_corners=False)
            pos = pos.permute(0, 2, 3, 1)
            if pos.shape[0] != B:
                pos = pos.repeat(B, 1, 1, 1)
            x_l = x_l + pos
            x_e = x_e + pos

        skips = []

        # ... Encoder Loop ...
        for i, blocks in enumerate(self.enc_stages):
            if verbose: print(f"Stage {i} Input: {x_l.shape}")
            for blk in blocks:
                x_l, x_e, x_fused = blk(x_l, x_e)

            skips.append(x_fused)

            if i < self.num_enc_layers - 1:
                x_l = self.downsamples_l[i](x_l)
                x_e = self.downsamples_e[i](x_e)

        # ... Bottleneck ...
        _, _, x_fused = self.bottleneck(x_l, x_e)
        if verbose: print(f"Bottleneck Out: {x_fused.shape}")

        # ... Decoder Loop ...
        for i, (up, reduce, layer) in enumerate(zip(self.upsamples, self.reduce_channels, self.decoder_layers)):
            idx = self.num_enc_layers - 2 - i

            x_up = up(x_fused)
            skip = skips[idx]

            if verbose: print(f"Decoder {i}: Up={x_up.shape}, Skip={skip.shape}")

            x_cat = torch.cat([x_up, skip], dim=-1)
            x_fused = reduce(x_cat)

            # 这里的 layer 现在是 nn.Sequential，会自动执行里面的所有 blocks
            x_fused = layer(x_fused)

            # ... Head ...
        x_fused = x_fused.permute(0, 3, 1, 2)

        # 2. 第一层上采样 + Norm + Act
        out = self.head_up1(x_fused)  # 输出 NCHW
        out = self.head_norm1(out)  # LayerNorm2d 接收 NCHW，输出 NCHW
        out = F.gelu(out)

        # 3. 第二层上采样 + Norm + Act
        out = self.head_up2(out)  # 输出 NCHW
        out = self.head_norm2(out)  # LayerNorm2d 接收 NCHW，输出 NCHW
        out = F.gelu(out)

        # 4. 最后一层卷积
        out = self.head_last(out)  # 输出 NCHW

        # Unpadding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Running Fixed SAMA-FusionMamba on {device}...")

    # [验证] 这里 decoder_depths=[2, 2, 2] 会在网络构建时被打印出来或者生效
    model = LEDepth(
        lidar_chans=1, event_chans=4, out_chans=1,
        patch_size=4, dims=96,
        depths=[2, 2, 6, 2],
        decoder_depths=[2, 2, 2]
    ).to(device)

    h, w = 720, 1280
    x_lidar = torch.randn(1, 1, h, w).to(device)
    x_event = torch.randn(1, 4, h, w).to(device)

    try:
        with torch.no_grad():
            out = model(x_lidar, x_event, verbose=True)
        print(f"✅ Pass! Output: {out.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()