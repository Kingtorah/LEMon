import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from submodules.vmamba import VSSM, VSSBlock, LayerNorm, Permute
except ImportError:
    try:
        from models.submodules.vmamba import VSSM, VSSBlock, LayerNorm, Permute
    except ImportError:
        print("")
        exit()


# =========================================================================
# 组件 0: LayerNorm2d替换vmamba.py的LayerNorm
# =========================================================================
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
# 组件 1: PatchExpand
# =========================================================================
class PatchExpand(nn.Module):
    def __init__(self, dim, scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.expand = nn.Linear(dim, (dim // 2) * (scale ** 2), bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        x = self.expand(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.scale)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


# =========================================================================
# 组件 2: FusionVMambaUNet
# =========================================================================
class LEDepth(nn.Module):
    def __init__(
            self,
            lidar_chans=1, event_chans=4, out_chans=1,
            patch_size=4,
            depths=[2, 2, 9, 2],
            dims=96,
            decoder_depths=[2, 2, 2],
            ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
            ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0",
            forward_type="v05_noz",
            mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0,
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

        # Encoder Components
        self.lidar_patch_embed = VSSM._make_patch_embed(lidar_chans, self.dims[0], patch_size, patch_norm, version="v1")
        self.event_patch_embed = VSSM._make_patch_embed(event_chans, self.dims[0], patch_size, patch_norm, version="v1")
        self.pos_embed = VSSM._pos_embed(self.dims[0], patch_size, imgsize)

        self.lidar_layers = nn.ModuleList()
        self.event_layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        for i in range(self.num_enc_layers):
            downsample = VSSM._make_downsample(
                self.dims[i], self.dims[i + 1], version="v1"
            ) if (i < self.num_enc_layers - 1) else nn.Identity()

            self.lidar_layers.append(self._make_vss_layer(
                dim=self.dims[i], depth=depths[i], downsample=downsample,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, ssm_init=ssm_init, forward_type=forward_type,
                ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer, use_checkpoint=use_checkpoint
            ))

            downsample_evt = VSSM._make_downsample(
                self.dims[i], self.dims[i + 1], version="v1"
            ) if (i < self.num_enc_layers - 1) else nn.Identity()

            self.event_layers.append(self._make_vss_layer(
                dim=self.dims[i], depth=depths[i], downsample=downsample_evt,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, ssm_init=ssm_init, forward_type=forward_type,
                ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer, use_checkpoint=use_checkpoint
            ))

        # Bottleneck
        current_dim = self.dims[-1] * 2
        self.bottleneck_reduce = nn.Linear(current_dim, self.dims[-1])
        self.bottleneck_block = self._make_vss_layer(
            dim=self.dims[-1], depth=2, downsample=nn.Identity(), drop_path=[0.0] * 2,
            ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, ssm_init=ssm_init, forward_type=forward_type,
            ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer
        )

        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.reduce_channels = nn.ModuleList()

        for i in range(self.num_enc_layers - 2, -1, -1):
            self.upsamples.append(PatchExpand(dim=self.dims[i + 1], scale=2))
            self.reduce_channels.append(nn.Linear(self.dims[i] * 3, self.dims[i]))
            self.decoder_layers.append(self._make_vss_layer(
                dim=self.dims[i], depth=decoder_depths[i], downsample=nn.Identity(),
                drop_path=[0.0] * decoder_depths[i],
                ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, ssm_init=ssm_init, forward_type=forward_type,
                ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer
            ))

        # Output Head
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(self.dims[0], self.dims[0] // 2, kernel_size=2, stride=2),
            LayerNorm2d(self.dims[0] // 2),
            nn.GELU(),
            nn.ConvTranspose2d(self.dims[0] // 2, 32, kernel_size=2, stride=2),
            LayerNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, out_chans, kernel_size=3, padding=1),
        )

        self.apply(VSSM(depths=[2])._init_weights)
        self.saved_lidar = None

    def _make_vss_layer(self, dim, depth, downsample, drop_path, **kwargs):
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU, sigmoid=nn.Sigmoid)
        if isinstance(kwargs.get('ssm_act_layer'), str):
            kwargs['ssm_act_layer'] = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), nn.SiLU)
        if isinstance(kwargs.get('mlp_act_layer'), str):
            kwargs['mlp_act_layer'] = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), nn.GELU)

        blocks = nn.ModuleList()
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, drop_path=drop_path[d] if isinstance(drop_path, list) else drop_path,
                channel_first=False, **kwargs
            ))
        return nn.ModuleDict({'blocks': blocks, 'downsample': downsample})

    def forward(self, x_lidar, x_event, verbose=False):
        if x_lidar is None:
            x_lidar = self.saved_lidar.clone()
        else:
            self.saved_lidar = x_lidar.clone()
        B, C, H, W = x_lidar.shape
        # verbose=True 时会打印形状
        if verbose:
            print(f"\n{'=' * 20} 前向传播开始 {'=' * 20}")
            print(f"1. 原始输入分辨率: {H} x {W}")
            print(f"   (Padding前) Lidar: {x_lidar.shape}, Event: {x_event.shape}")

        # Step A: 自动 Padding
        factor = self.total_factor
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        if pad_h > 0 or pad_w > 0:
            x_lidar = F.pad(x_lidar, (0, pad_w, 0, pad_h))
            x_event = F.pad(x_event, (0, pad_w, 0, pad_h))
            if verbose: print(f"   [Padding] 为了适配倍率{factor}，填充为: {x_lidar.shape}")

        # Step B: Patch & Pos Embed
        x_l = self.lidar_patch_embed(x_lidar)
        x_e = self.event_patch_embed(x_event)

        if verbose:
            print(f"\n2. Patch Embedding (下采样 {self.patch_size} 倍)")
            print(f"   特征图大小: {x_l.shape} (H/{self.patch_size}, W/{self.patch_size})")
            print(f"   通道数(Dim): {x_l.shape[-1]} (Stage 0)")

        if self.pos_embed is not None:
            curr_h, curr_w = x_l.shape[1], x_l.shape[2]
            pos_emb = self.pos_embed
            if pos_emb.shape[2] != curr_h or pos_emb.shape[3] != curr_w:
                pos_emb = F.interpolate(pos_emb, size=(curr_h, curr_w), mode='bilinear', align_corners=False)
            x_l = x_l + pos_emb.permute(0, 2, 3, 1)
            x_e = x_e + pos_emb.permute(0, 2, 3, 1)

        # Step C: Encoder
        if verbose: print(f"\n3. Encoder (编码器下采样流程)")
        skips_l = []
        skips_e = []
        for i, (l_layer, e_layer) in enumerate(zip(self.lidar_layers, self.event_layers)):
            # Process Blocks
            for blk in l_layer['blocks']: x_l = blk(x_l)
            for blk in e_layer['blocks']: x_e = blk(x_e)

            # Save Skips
            skips_l.append(x_l)
            skips_e.append(x_e)

            if verbose:
                print(f"   [Stage {i}] 输出形状: {x_l.shape} -> 保存为 Skip Connection")

            # Downsample (Go to next stage)
            x_l = l_layer['downsample'](x_l)
            x_e = e_layer['downsample'](x_e)

            if verbose and i < self.num_enc_layers - 1:
                print(f"       ↓ 下采样 ↓ 到: {x_l.shape}")

        # Step D: Fusion
        if verbose: print(f"\n4. Bottleneck Fusion (瓶颈层)")
        x_fused = torch.cat([x_l, x_e], dim=-1)
        if verbose: print(f"   拼接 (Concat): {x_fused.shape} (Lidar+Event)")

        x_fused = self.bottleneck_reduce(x_fused)
        if verbose: print(f"   降维 (Reduce): {x_fused.shape}")

        for blk in self.bottleneck_block['blocks']: x_fused = blk(x_fused)

        # Step E: Decoder
        if verbose: print(f"\n5. Decoder (解码器上采样流程)")
        for i, (upsample, reduce, layer) in enumerate(zip(self.upsamples, self.reduce_channels, self.decoder_layers)):
            skip_idx = self.num_enc_layers - 2 - i

            # Upsample
            x_prev_shape = x_fused.shape
            x_up = upsample(x_fused)
            if verbose: print(f"   [Decoder Stage {skip_idx}] 上采样: {x_prev_shape} -> {x_up.shape}")

            # Get Skips
            s_l = skips_l[skip_idx]
            s_e = skips_e[skip_idx]
            if verbose: print(f"       + 跳跃连接 Lidar: {s_l.shape}")
            if verbose: print(f"       + 跳跃连接 Event: {s_e.shape}")

            # Concat & Reduce
            x_cat = torch.cat([x_up, s_l, s_e], dim=-1)
            x_fused = reduce(x_cat)
            if verbose: print(f"       = 融合后形状: {x_fused.shape}")

            # Process
            for blk in layer['blocks']: x_fused = blk(x_fused)

        # Step F: Head
        if verbose: print(f"\n6. Final Head (恢复原分辨率)")
        x_fused = x_fused.permute(0, 3, 1, 2)
        if verbose: print(f"   Permute后: {x_fused.shape}")

        out = self.final_up(x_fused)
        if verbose: print(f"   Final Up 输出: {out.shape}")

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
            if verbose: print(f"   [Unpadding] 切割回原尺寸: {out.shape}")

        if verbose: print(f"{'=' * 20} 前向传播结束 {'=' * 20}\n")

        return out


# =========================================================================
# 测试脚本 (Main)
# =========================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...")

    # 1. 实例化模型 (Tiny 配置)
    # patch_size=4, dims=96
    model = LEDepth(
        lidar_chans=1, event_chans=4, out_chans=1,
        patch_size=4,
        dims=96,  #  [96, 192, 384, 768]
        depths=[2, 2, 9, 2],
        decoder_depths=[2, 2, 2],
    ).to(device)

    # 2. 测试输入: 720p 分辨率 (720x1280)
    # 这是一个很好的测试用例，因为 720 不是 32 的倍数，可以测试 Padding
    h, w = 720, 1280
    x_lidar = torch.randn(1, 1, h, w).to(device)
    x_event = torch.randn(1, 4, h, w).to(device)

    # 3. 开启 verbose=True 进行详细打印
    with torch.no_grad():
        out = model(x_lidar, x_event, verbose=True)

    # 4. 验证结果
    if out.shape[-2:] == (h, w):
        print("✅ 测试通过！最终输出分辨率与输入一致。")
    else:
        print("❌ 测试失败，尺寸不匹配。")


if __name__ == "__main__":
    main()