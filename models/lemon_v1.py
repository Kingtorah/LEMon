import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from submodules.vmamba import VSSM, VSSBlock, LayerNorm, Permute
except ImportError:
    try:
        from models.submodules.vmamba import VSSM, VSSBlock, LayerNorm, Permute
    except ImportError:
        print("【严重错误】找不到 vmamba.py，请检查路径。")
        exit()


# =========================================================================
# 组件 0: LayerNorm2d (修复版：专门处理 NCHW 格式)
# =========================================================================
class LayerNorm2d(nn.Module):
    """
    适用于 (B, C, H, W) 格式输入的 LayerNorm
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # x: (B, C, H, W)
        x = x.permute(0, 2, 3, 1)  # -> (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # -> (B, C, H, W)
        return x


# =========================================================================
# 组件 1: PatchExpand (NHWC 格式)
# =========================================================================
class PatchExpand(nn.Module):
    def __init__(self, dim, scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.expand = nn.Linear(dim, (dim // 2) * (scale ** 2), bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        # Input: (B, H, W, C)
        x = self.expand(x)

        # Pixel Shuffle 需要 NCHW
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.scale)
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)
        return x  # Output: (B, H*s, W*s, C/2)


# =========================================================================
# 组件 2: CrossMambaFusion (NHWC 格式)
# =========================================================================
class CrossMambaFusion(nn.Module):
    def __init__(self, dim, ssm_kwargs):
        super().__init__()
        self.reduce = nn.Linear(dim * 2, dim)

        # [注意] 这里处理的是 NHWC 格式，直接用 nn.LayerNorm 即可
        self.norm = nn.LayerNorm(dim)

        self.fusion_mamba = VSSBlock(hidden_dim=dim, **ssm_kwargs)

        self.gate_conv = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x_l, x_e):
        # 输入期望: (B, H, W, C)

        # A. 拼接
        x_cat = torch.cat([x_l, x_e], dim=-1)  # (B, H, W, 2C)
        x_merged = self.reduce(x_cat)
        x_merged = self.norm(x_merged)

        # B. Mamba 上下文
        x_context = self.fusion_mamba(x_merged)

        # C. 门控
        gate = self.gate_conv(x_context)

        # D. 融合
        x_fused = gate * x_l + (1 - gate) * x_e + x_context

        return x_fused  # 输出: (B, H, W, C)


# =========================================================================
# 组件 3: LEDepth (主网络)
# =========================================================================
class LEDepth(nn.Module):
    def __init__(
            self,
            lidar_chans=1, event_chans=4, out_chans=1,
            patch_size=4,
            depths=[2, 2, 9, 2],
            dims=96,
            decoder_depths=[2, 2, 2],
            ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0",
            forward_type="v05_noz",
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            drop_path_rate=0.2, patch_norm=True, use_checkpoint=False,
            imgsize=224,
            **kwargs,
    ):
        super().__init__()

        self.num_enc_layers = len(depths)
        self.downsample_times = self.num_enc_layers - 1
        self.total_factor = patch_size * (2 ** self.downsample_times)

        if isinstance(dims, int):
            self.dims = [int(dims * 2 ** i) for i in range(self.num_enc_layers)]
        else:
            self.dims = dims

        self.patch_size = patch_size

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

        # 1. Embeddings (Output NHWC)
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

            if i < self.num_enc_layers - 1:
                self.fusion_blocks.append(CrossMambaFusion(self.dims[i], ssm_kwargs))

        # 2. Bottleneck
        self.bottleneck_fusion = CrossMambaFusion(self.dims[-1], ssm_kwargs)

        # 3. Decoder
        self.decoder_layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.reduce_channels = nn.ModuleList()

        for i in range(self.num_enc_layers - 2, -1, -1):
            self.upsamples.append(PatchExpand(dim=self.dims[i + 1], scale=2))
            self.reduce_channels.append(nn.Linear(self.dims[i] * 2, self.dims[i]))
            self.decoder_layers.append(self._make_vss_layer(
                dim=self.dims[i], depth=decoder_depths[i], downsample=nn.Identity(),
                drop_path=[0.0] * decoder_depths[i],
                use_checkpoint=use_checkpoint, **ssm_kwargs
            ))

        # 4. Output Head
        self.head_up1 = nn.ConvTranspose2d(self.dims[0], self.dims[0] // 2, kernel_size=2, stride=2)
        self.head_norm1 = LayerNorm2d(self.dims[0] // 2)
        self.head_up2 = nn.ConvTranspose2d(self.dims[0] // 2, 32, kernel_size=2, stride=2)
        self.head_norm2 = LayerNorm2d(32)
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
        if x_lidar is None:
            if self.saved_lidar is not None:
                x_lidar = self.saved_lidar.clone()
            else:
                B, _, H, W = x_event.shape
                x_lidar = torch.zeros((B, 1, H, W), device=x_event.device, dtype=x_event.dtype)
        else:
            self.saved_lidar = x_lidar.clone()

        # Validity Mask
        valid_mask = (x_lidar > 0).float()
        x_lidar_in = torch.cat([x_lidar, valid_mask], dim=1)

        # Padding
        B, _, H, W = x_lidar.shape
        factor = self.total_factor
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        if pad_h > 0 or pad_w > 0:
            x_lidar_in = F.pad(x_lidar_in, (0, pad_w, 0, pad_h))
            x_event = F.pad(x_event, (0, pad_w, 0, pad_h))

        # Embed (-> NHWC)
        x_l = self.lidar_patch_embed(x_lidar_in)
        x_e = self.event_patch_embed(x_event)

        # Pos Embed (NCHW -> NHWC)
        if self.pos_embed is not None:
            curr_h, curr_w = x_l.shape[1], x_l.shape[2]
            pos = self.pos_embed
            if pos.shape[2] != curr_h or pos.shape[3] != curr_w:
                pos = F.interpolate(pos, size=(curr_h, curr_w), mode='bilinear', align_corners=False)
            pos = pos.permute(0, 2, 3, 1)  # NCHW -> NHWC
            x_l = x_l + pos
            x_e = x_e + pos

        # Encoder
        skips_fused = []

        for i, (l_layer, e_layer) in enumerate(zip(self.lidar_layers, self.event_layers)):
            for blk in l_layer['blocks']: x_l = blk(x_l)
            for blk in e_layer['blocks']: x_e = blk(x_e)

            # Fusion
            if i < len(self.fusion_blocks):
                # CrossMambaFusion 输入输出都是 NHWC
                fused = self.fusion_blocks[i](x_l, x_e)
                skips_fused.append(fused)

            # Downsample
            x_l = l_layer['downsample'](x_l)
            x_e = e_layer['downsample'](x_e)

        # Bottleneck
        x_fused = self.bottleneck_fusion(x_l, x_e)  # NHWC

        # Decoder
        for i, (upsample, reduce, layer) in enumerate(zip(self.upsamples, self.reduce_channels, self.decoder_layers)):
            idx = self.num_enc_layers - 2 - i

            # Upsample
            x_up = upsample(x_fused)  # NHWC

            skip = skips_fused[idx]  # NHWC

            # Concat
            x_cat = torch.cat([x_up, skip], dim=-1)
            x_fused = reduce(x_cat)

            for blk in layer['blocks']: x_fused = blk(x_fused)

        # Head (转换回 NCHW 进行卷积处理)
        x_fused = x_fused.permute(0, 3, 1, 2)  # NHWC -> NCHW

        out = self.head_up1(x_fused)
        out = self.head_norm1(out)  # LayerNorm2d 正常工作 (NCHW -> NHWC -> Norm -> NCHW)
        out = F.gelu(out)

        out = self.head_up2(out)
        out = self.head_norm2(out)  # LayerNorm2d 正常工作
        out = F.gelu(out)

        out = self.head_last(out)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out


# =========================================================================
# 测试脚本
# =========================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...")

    model = LEDepth(
        lidar_chans=1, event_chans=4, out_chans=1,
        patch_size=4, dims=96
    ).to(device)

    h, w = 720, 1280
    x_lidar = torch.randn(1, 1, h, w).to(device)
    x_event = torch.randn(1, 4, h, w).to(device)

    with torch.no_grad():
        out = model(x_lidar, x_event, verbose=True)

    print(f"输出尺寸: {out.shape}")
    if out.shape[-2:] == (h, w):
        print("✅ 测试通过！")
    else:
        print("❌ 尺寸不匹配。")


if __name__ == "__main__":
    main()