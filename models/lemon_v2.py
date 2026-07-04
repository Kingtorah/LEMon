import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# 尝试导入基础组件
try:
    from submodules.vmamba import VSSM, VSSBlock, LayerNorm, Permute
except ImportError:
    try:
        from models.submodules.vmamba import VSSM, VSSBlock, LayerNorm, Permute
    except ImportError:
        print("【严重错误】找不到 vmamba.py，请检查路径。")
        exit()


# =========================================================================
# 组件 1: 亲和力生成器 (Affinity Generator)
# =========================================================================
class AffinityGenerator(nn.Module):
    """
    利用 Event 数据生成 Affinity Map (亲和力图)
    Affinity 越高，表示像素间越相似（平滑），信息越应该流动。
    """

    def __init__(self, dim):
        super().__init__()
        # 这是一个轻量级的 U-Net like 结构或简单的 Conv 堆叠
        # 用于从 Event 特征中提取边缘/纹理信息，并转化为亲和力
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim // 2),
            nn.SiLU(),
            nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出 0~1 之间的亲和力系数
        )

    def forward(self, x_event):
        # 输入: Event Feature (B, C, H, W)
        # 输出: Affinity Map (B, C, H, W)
        # 1.0 代表完全亲和 (平滑)，0.0 代表完全阻断 (边缘)
        return self.net(x_event)


# =========================================================================
# 组件 2: 置信度估计器 (Confidence Estimator)
# =========================================================================
class ConfidenceEstimator(nn.Module):
    """
    评估当前特征的可靠性。
    LiDAR 的非零区域天然具有高置信度。
    Event 的噪声区域具有低置信度。
    """

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),  # 输入是 Concat(LiDAR, Event)
            nn.SiLU(),
            nn.Conv2d(dim, 1, kernel_size=1),  # 输出单通道置信度图
            nn.Sigmoid()
        )

    def forward(self, x_lidar, x_event):
        # 融合两者信息来判断置信度
        cat = torch.cat([x_lidar, x_event], dim=1)
        conf = self.net(cat)
        return conf  # (B, 1, H, W)


# =========================================================================
# 组件 3: AC-Fusion Block (核心创新)
# =========================================================================
class AC_FusionBlock(nn.Module):
    """
    Affinity-Confidence Fusion Block
    逻辑：
    1. Affinity 决定 Mamba 内部的信息流动 (Modulate Input/State)
    2. Confidence 决定最终输出的残差权重 (Hard Constraint)
    """

    def __init__(self, dim, ssm_kwargs):
        super().__init__()
        self.dim = dim

        # 1. 两个辅助分支
        self.affinity_net = AffinityGenerator(dim)
        self.confidence_net = ConfidenceEstimator(dim)

        # 2. 特征预融合
        self.pre_fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.LayerNorm([dim, 1, 1]) if hasattr(nn, "LayerNorm") else nn.Identity()
        )

        # 3. Mamba 核心 (用于长距离传播)
        self.mamba = VSSBlock(hidden_dim=dim, **ssm_kwargs)

        # 4. 亲和力调制层
        # 用 Affinity 来缩放特征，模拟"由亲和力控制的扩散"
        self.affinity_proj = nn.Linear(dim, dim)

    def forward(self, x_l, x_e):
        # x_l: LiDAR feature (B, C, H, W)
        # x_e: Event feature (B, C, H, W)

        # --- A. 生成辅助图 ---
        # 1. Affinity Map (B, C, H, W)
        # Event 越平滑 -> Affinity 越高 -> 信息流动越快
        affinity = self.affinity_net(x_e)

        # 2. Confidence Map (B, 1, H, W)
        # 我们希望网络学会：LiDAR 有值的地方 Conf=1，无值的地方 Conf由网络预测确定
        confidence = self.confidence_net(x_l, x_e)

        # --- B. 亲和力调制的特征融合 ---
        # 这里的创新点：
        # 我们不直接把 x_l 扔进 Mamba，而是先用 Affinity 进行加权
        # 物理意义：如果 Affinity 低（边缘），我们抑制该处的特征幅度，防止它在 Mamba 扫描时过度扩散

        # 转换到 NHWC 进行线性操作
        x_l_perm = x_l.permute(0, 2, 3, 1)
        aff_perm = affinity.permute(0, 2, 3, 1)

        # Affinity Modulation: I_modulated = I * A
        # 这是一种软性的"阻断"，边缘处的特征值变小，经过 SSM 积分时贡献就变小
        x_l_modulated = x_l_perm * aff_perm
        x_l_modulated = x_l_modulated.permute(0, 3, 1, 2)

        # 预融合
        x_merged = self.pre_fusion(torch.cat([x_l_modulated, x_e], dim=1))
        x_merged = x_merged.permute(0, 2, 3, 1)  # -> NHWC

        # --- C. Mamba 传播 (Propagation) ---
        # Mamba 在这里充当了一个高效的"各向异性扩散求解器"
        # 经过调制的特征进入 Mamba，会自动在 Affinity 高的区域平滑，在 Affinity 低的区域截止
        x_prop = self.mamba(x_merged)  # (B, H, W, C)
        x_prop = x_prop.permute(0, 3, 1, 2)  # -> NCHW

        # --- D. 基于置信度的残差校正 (Confidence Rectification) ---
        # 这是很多 Depth Completion SOTA 方法（如 NLSPN）的核心思想
        # y = Conf * x_raw + (1 - Conf) * x_prop
        #
        # 但这里要注意：x_l 是特征层，不是原始深度图，所以我们用一种"软校正"
        # 如果 Confidence 高，我们更倾向于保留输入特征中的原始信息（通过残差）
        # 如果 Confidence 低（空洞），我们完全依赖 Mamba 传播来的信息

        # 残差连接：x_l 是原始 LiDAR 特征
        out = confidence * x_l + (1 - confidence) * x_prop

        return out


# =========================================================================
# 组件 4: FusionMamba (主网络架构)
# =========================================================================
class FusionMamba(nn.Module):
    def __init__(
            self,
            lidar_chans=1, event_chans=4, out_chans=1,
            patch_size=4,
            depths=[2, 2, 9, 2],
            dims=96,
            decoder_depths=[2, 2, 2],
            ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto",
            ssm_act_layer="silu", ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
            ssm_init="v0", forward_type="v05_noz",
            mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0,
            drop_path_rate=0.2, patch_norm=True, use_checkpoint=False,
            imgsize=224,
            **kwargs,
    ):
        super().__init__()

        # ... (常规参数初始化逻辑同前) ...
        self.num_enc_layers = len(depths)
        if isinstance(dims, int):
            self.dims = [int(dims * 2 ** i) for i in range(self.num_enc_layers)]
        else:
            self.dims = dims
        self.patch_size = patch_size
        self.total_factor = patch_size * (2 ** (self.num_enc_layers - 1))

        # 激活函数转换
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU, sigmoid=nn.Sigmoid)
        if isinstance(ssm_act_layer, str): ssm_act_layer = _ACTLAYERS.get(ssm_act_layer.lower(), nn.SiLU)
        if isinstance(mlp_act_layer, str): mlp_act_layer = _ACTLAYERS.get(mlp_act_layer.lower(), nn.GELU)

        ssm_kwargs = dict(
            ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer, ssm_conv=ssm_conv, ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate, ssm_init=ssm_init, forward_type=forward_type,
            mlp_ratio=mlp_ratio, mlp_act_layer=mlp_act_layer, mlp_drop_rate=mlp_drop_rate,
            channel_first=False
        )

        # 1. Embedding
        # 依然使用 Validity Mask 策略，因为它符合 Confidence 的逻辑
        self.lidar_patch_embed = VSSM._make_patch_embed(lidar_chans + 1, self.dims[0], patch_size, patch_norm,
                                                        version="v1")
        self.event_patch_embed = VSSM._make_patch_embed(event_chans, self.dims[0], patch_size, patch_norm, version="v1")
        self.pos_embed = VSSM._pos_embed(self.dims[0], patch_size, imgsize)

        self.lidar_layers = nn.ModuleList()
        self.event_layers = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        for i in range(self.num_enc_layers):
            ds = VSSM._make_downsample(self.dims[i], self.dims[i + 1],
                                       version="v1") if i < self.num_enc_layers - 1 else nn.Identity()
            ds_evt = VSSM._make_downsample(self.dims[i], self.dims[i + 1],
                                           version="v1") if i < self.num_enc_layers - 1 else nn.Identity()

            self.lidar_layers.append(self._make_vss_layer(
                dim=self.dims[i], depth=depths[i], downsample=ds,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                use_checkpoint=use_checkpoint, **ssm_kwargs
            ))

            self.event_layers.append(self._make_vss_layer(
                dim=self.dims[i], depth=depths[i], downsample=ds_evt,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                use_checkpoint=use_checkpoint, **ssm_kwargs
            ))

            # [NEW] 使用 AC_FusionBlock 替代 CrossMambaFusion
            if i < self.num_enc_layers - 1:
                self.fusion_blocks.append(AC_FusionBlock(self.dims[i], ssm_kwargs))

        # 2. Bottleneck
        self.bottleneck_fusion = AC_FusionBlock(self.dims[-1], ssm_kwargs)

        # 3. Decoder
        self.decoder_layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.reduce_channels = nn.ModuleList()

        # PatchExpand (NCHW version needed)
        class PatchExpand(nn.Module):
            def __init__(self, dim, scale=2):
                super().__init__()
                self.expand = nn.Linear(dim, (dim // 2) * scale * scale, bias=False)
                self.norm = nn.LayerNorm(dim // 2)
                self.scale = scale

            def forward(self, x):
                # Input NCHW -> NHWC
                x = x.permute(0, 2, 3, 1)
                x = self.expand(x)
                x = x.permute(0, 3, 1, 2)
                x = F.pixel_shuffle(x, self.scale)
                x = x.permute(0, 2, 3, 1)
                x = self.norm(x)
                return x.permute(0, 3, 1, 2)

        for i in range(self.num_enc_layers - 2, -1, -1):
            self.upsamples.append(PatchExpand(dim=self.dims[i + 1], scale=2))
            self.reduce_channels.append(nn.Linear(self.dims[i] * 2, self.dims[i]))
            self.decoder_layers.append(self._make_vss_layer(
                dim=self.dims[i], depth=decoder_depths[i], downsample=nn.Identity(),
                drop_path=[0.0] * decoder_depths[i],
                use_checkpoint=use_checkpoint, **ssm_kwargs
            ))

        # 4. Head
        # Output Head NCHW
        self.head_up1 = nn.ConvTranspose2d(self.dims[0], self.dims[0] // 2, kernel_size=2, stride=2)
        self.head_norm1 = nn.LayerNorm([self.dims[0] // 2, 1, 1])  # Placeholder, adjusted in forward
        self.head_up2 = nn.ConvTranspose2d(self.dims[0] // 2, 32, kernel_size=2, stride=2)
        self.head_norm2 = nn.LayerNorm([32, 1, 1])
        self.head_last = nn.Conv2d(32, out_chans, kernel_size=3, padding=1)

        self.apply(VSSM(depths=[2])._init_weights)
        self.saved_lidar = None

    def _make_vss_layer(self, dim, depth, downsample, drop_path, **kwargs):
        blocks = nn.ModuleList()
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d] if isinstance(drop_path, list) else drop_path,
                **kwargs
            ))
        return nn.ModuleDict({'blocks': blocks, 'downsample': downsample})

    def forward(self, x_lidar, x_event, verbose=False):
        # ... (常规预处理：Zero Hold, Mask, Padding) ...
        # (代码与之前相同，为了节省长度省略，重点在 AC-Fusion 的调用)
        if x_lidar is None:
            if self.saved_lidar is not None:
                x_lidar = self.saved_lidar.clone()
            else:
                B, _, H, W = x_event.shape
                x_lidar = torch.zeros((B, 1, H, W), device=x_event.device, dtype=x_event.dtype)
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

        x_l = self.lidar_patch_embed(x_lidar_in)  # NHWC
        x_e = self.event_patch_embed(x_event)

        # Pos Embed (NCHW -> NHWC fix)
        if self.pos_embed is not None:
            pos = self.pos_embed
            if pos.shape[2] != x_l.shape[1] or pos.shape[3] != x_l.shape[2]:
                pos = F.interpolate(pos, size=(x_l.shape[1], x_l.shape[2]), mode='bilinear', align_corners=False)
            pos = pos.permute(0, 2, 3, 1)
            x_l = x_l + pos
            x_e = x_e + pos

        # 恢复成 NCHW 进行处理，因为 AC_FusionBlock 内部处理 Conv 方便
        x_l = x_l.permute(0, 3, 1, 2)
        x_e = x_e.permute(0, 3, 1, 2)

        skips_fused = []

        for i, (l_layer, e_layer) in enumerate(zip(self.lidar_layers, self.event_layers)):
            # VSSBlock 输入要求 NHWC，需要转换
            x_l_perm = x_l.permute(0, 2, 3, 1)
            x_e_perm = x_e.permute(0, 2, 3, 1)

            for blk in l_layer['blocks']: x_l_perm = blk(x_l_perm)
            for blk in e_layer['blocks']: x_e_perm = blk(x_e_perm)

            # 转回 NCHW
            x_l = x_l_perm.permute(0, 3, 1, 2)
            x_e = x_e_perm.permute(0, 3, 1, 2)

            # [Fusion]
            if i < len(self.fusion_blocks):
                # AC_FusionBlock 输入要求 NCHW (为了方便 Conv)
                fused = self.fusion_blocks[i](x_l, x_e)
                skips_fused.append(fused)  # 存储 NCHW

            # Downsample
            x_l_perm = x_l.permute(0, 2, 3, 1)
            x_e_perm = x_e.permute(0, 2, 3, 1)
            x_l_perm = l_layer['downsample'](x_l_perm)
            x_e_perm = e_layer['downsample'](x_e_perm)
            x_l = x_l_perm.permute(0, 3, 1, 2)
            x_e = x_e_perm.permute(0, 3, 1, 2)

        # Bottleneck
        x_fused = self.bottleneck_fusion(x_l, x_e)  # NCHW

        # Decoder
        for i, (upsample, reduce, layer) in enumerate(zip(self.upsamples, self.reduce_channels, self.decoder_layers)):
            idx = self.num_enc_layers - 2 - i

            x_up = upsample(x_fused)  # NCHW
            skip = skips_fused[idx]  # NCHW

            x_cat = torch.cat([x_up, skip], dim=1)  # Channel concat

            # Reduce & Block (需要 NHWC)
            x_cat_perm = x_cat.permute(0, 2, 3, 1)
            x_fused_perm = reduce(x_cat_perm)
            for blk in layer['blocks']: x_fused_perm = blk(x_fused_perm)
            x_fused = x_fused_perm.permute(0, 3, 1, 2)

        # Head
        out = self.head_up1(x_fused)
        # 手动 Norm 避免维度问题
        out = out.permute(0, 2, 3, 1)
        out = F.layer_norm(out, [self.dims[0] // 2])
        out = out.permute(0, 3, 1, 2)
        out = F.gelu(out)

        out = self.head_up2(out)
        out = out.permute(0, 2, 3, 1)
        out = F.layer_norm(out, [32])
        out = out.permute(0, 3, 1, 2)
        out = F.gelu(out)

        out = self.head_last(out)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out