import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

# 引用同目录下的 submodules
from models.submodules.ledepth_submodules import ConvEncodingHead, ConvDecodingHead, VSSBlock


class LEDepth(nn.Module):
    """
    HR-VSSN: Hierarchical Recurrent Visual State Space Network
    (Mamba-DELTA)

    特点:
    1. 使用 SSM 替代 Transformer，线性复杂度，更省显存。
    2. 全层级时序记忆 (new_states)。
    3. 支持任意分辨率 (Auto Padding + Dynamic Pos Embed)。
    """

    def __init__(self,
                 lidar_channels: int = 1,
                 event_channels: int = 1,
                 out_channels: int = 1,
                 num_layers: int = 4,  # 深度
                 patch_size: int = 4,  # Patch 大小
                 dimensionality: int = 96,  # 特征维度
                 d_state: int = 16):  # SSM 记忆维度
        super(LEDepth, self).__init__()

        self.patch_size = patch_size
        self.num_layers = num_layers
        self.dimensionality = dimensionality

        # --- 1. 编码头 ---
        self.lidar_head = ConvEncodingHead(patch_size, lidar_channels, dimensionality)
        self.event_head = ConvEncodingHead(patch_size, event_channels, dimensionality)

        # --- 2. 动态位置编码 ---
        # 定义一个基础尺寸 (比如基于训练时的 256x256)，forward 时会动态调整
        base_h, base_w = 256 // patch_size, 256 // patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, dimensionality, base_h, base_w))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- 3. Mamba 编码器 (双流) ---
        # LiDAR 分支
        self.lidar_ssm_layers = nn.ModuleList([
            VSSBlock(dimensionality, d_state) for _ in range(num_layers)
        ])
        # Event 分支
        self.event_ssm_layers = nn.ModuleList([
            VSSBlock(dimensionality, d_state) for _ in range(num_layers)
        ])

        # --- 4. 门控融合模块 ---
        self.fusion_gate = nn.Sequential(
            nn.Linear(dimensionality * 2, dimensionality),
            nn.Sigmoid()
        )
        self.fusion_ssm = VSSBlock(dimensionality, d_state)

        # --- 5. 解码器 ---
        self.decoder_ssm_layers = nn.ModuleList([
            VSSBlock(dimensionality, d_state) for _ in range(num_layers)
        ])
        self.skip_norm = nn.ModuleList([nn.LayerNorm(dimensionality) for _ in range(num_layers)])
        self.decoding_head = ConvDecodingHead(dimensionality, patch_size, out_channels)

    def resize_pos_embed(self, posemb, new_h, new_w):
        """
        核心功能：动态调整位置编码大小以适应任意分辨率
        """
        # 如果尺寸一致，直接返回
        if posemb.shape[-2:] == (new_h, new_w):
            return posemb.flatten(2).transpose(1, 2)

        # 双线性插值调整
        posemb_new = F.interpolate(
            posemb, size=(new_h, new_w), mode='bilinear', align_corners=False
        )
        # (1, D, H, W) -> (1, L, D)
        return posemb_new.flatten(2).transpose(1, 2)

    def forward(self,
                lidar_input: Tensor | None,
                event_input: Tensor,
                prev_states: dict | None = None) -> tuple[Tensor, dict]:
        """
        Forward Pass.
        prev_states: 包含上一帧所有层 hidden state 的字典。
        """

        # --- 0. 自动 Padding (处理不能整除的情况) ---
        # 例如输入 346x260, patch=4 -> 需要 Pad 到 348x260
        B, C, H, W = event_input.shape
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size

        if pad_h > 0 or pad_w > 0:
            event_input = F.pad(event_input, (0, pad_w, 0, pad_h))
            if lidar_input is not None:
                lidar_input = F.pad(lidar_input, (0, pad_w, 0, pad_h))

        # 获取新尺寸
        H_pad, W_pad = event_input.shape[-2:]
        h_p, w_p = H_pad // self.patch_size, W_pad // self.patch_size

        # --- 1. 状态初始化 ---
        if lidar_input is None:
            # 简单的缺失处理，实际可复用上一帧 LiDAR
            lidar_input = torch.zeros_like(event_input[:, :1])

        if prev_states is None:
            prev_states = {
                'lidar_s': [None] * self.num_layers,
                'event_s': [None] * self.num_layers,
                'fusion_s': None,
                'decoder_s': [None] * self.num_layers
            }

        new_states = {'lidar_s': [], 'event_s': [], 'fusion_s': None, 'decoder_s': []}

        # --- 2. 编码与位置嵌入 ---
        l_feat, _ = self.lidar_head(lidar_input)
        e_feat, skip_evt_raw = self.event_head(event_input)  # skip_evt_raw 用于最终 head

        l_feat = l_feat.flatten(2).transpose(1, 2)  # (B, L, D)
        e_feat = e_feat.flatten(2).transpose(1, 2)

        # 动态位置编码
        pos = self.resize_pos_embed(self.pos_embed, h_p, w_p)
        l_feat = l_feat + pos
        e_feat = e_feat + pos

        # --- 3. 双流 SSM 编码 ---
        skip_connections = []

        for i in range(self.num_layers):
            # Event Branch
            e_feat, e_s_new = self.event_ssm_layers[i](e_feat, prev_states['event_s'][i])
            new_states['event_s'].append(e_s_new)

            # LiDAR Branch
            l_feat, l_s_new = self.lidar_ssm_layers[i](l_feat, prev_states['lidar_s'][i])
            new_states['lidar_s'].append(l_s_new)

            # 收集 Skip
            skip_connections.append(l_feat + e_feat)

        # --- 4. 融合 ---
        # 简单融合：Gate * LiDAR + Event
        cat_feat = torch.cat([l_feat, e_feat], dim=-1)
        gate = self.fusion_gate(cat_feat)
        fused_feat = (l_feat * gate) + (e_feat * (1 - gate))

        # 时序平滑融合
        central_feat, f_s_new = self.fusion_ssm(fused_feat, prev_states['fusion_s'])
        new_states['fusion_s'] = f_s_new

        # --- 5. SSM 解码 ---
        x = central_feat
        for i in range(self.num_layers):
            # 对称 Skip Connection
            skip = skip_connections[-(i + 1)]
            x = x + self.skip_norm[i](skip)

            x, d_s_new = self.decoder_ssm_layers[i](x, prev_states['decoder_s'][i])
            new_states['decoder_s'].append(d_s_new)

        # --- 6. 输出头与 Unpadding ---
        depth_pred = self.decoding_head(x, skip_evt_raw, h_p, w_p)

        # 切除 Padding 部分，恢复原始分辨率
        if pad_h > 0 or pad_w > 0:
            depth_pred = depth_pred[:, :, :H, :W]

        return depth_pred, new_states