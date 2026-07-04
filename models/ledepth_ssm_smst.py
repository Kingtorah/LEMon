import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from models.submodules.ledepth_submodules import ConvEncodingHead, ConvDecodingHead, VSSBlock, StructureGuidedBlock, \
    MultiHeadStructureGuidedBlock, ParallaxHead
from models.submodules.shared_submodules import PositionalEncoder2D


class LEDepth(nn.Module):
    """
    LEDepth (Specific-Task Design) with DELTA's Positional Encoding
    """

    def __init__(self,
                 lidar_channels: int = 1,
                 event_channels: int = 4,
                 out_channels: int = 1,
                 num_layers: int = 4,
                 patch_size: int = 4,
                 num_heads: int = 1,
                 dimensionality: int = 96,
                 d_state: int = 16,
                 max_res: tuple = (720, 1284)):  # 用于位置编码
        super(LEDepth, self).__init__()

        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dimensionality = dimensionality

        # --- 1. Encoders ---
        self.lidar_head = ConvEncodingHead(patch_size, lidar_channels, dimensionality)
        self.event_head = ConvEncodingHead(patch_size, event_channels, dimensionality)

        # --- Modified Positional Embed (Using DELTA's Logic) ---
        # 使用 DELTA 的 2D 位置编码器
        # max_res 默认为 (720, 1284)，对应 (height, width)
        self.pos_encoder = PositionalEncoder2D(dimensionality, (max_res[0] // patch_size, max_res[1] // patch_size))

        # --- 2. Independent Processing Streams ---
        self.lidar_ssm_layers = nn.ModuleList([
            VSSBlock(dimensionality, d_state) for _ in range(num_layers)
        ])

        self.event_ssm_layers = nn.ModuleList([
            VSSBlock(dimensionality, d_state) for _ in range(num_layers)
        ])

        # --- 3. [ANGLE 2] Teacher/Student Mechanism ---
        self.student_parallax_head = ParallaxHead(dimensionality, patch_size)

        # --- 4. [ANGLE 1] Master/Slave Densification Fusion ---
        self.densification_layers = nn.ModuleList([
            MultiHeadStructureGuidedBlock(dimensionality, num_heads, d_state) for _ in range(num_layers)
        ])

        # 融合后的 Mixer
        self.post_fusion_mix = nn.Sequential(
            nn.Linear(dimensionality, dimensionality),
            nn.GELU()
        )

        # --- 5. Decoder ---
        self.decoder_ssm_layers = nn.ModuleList([
            VSSBlock(dimensionality, d_state) for _ in range(num_layers)
        ])
        self.skip_norm = nn.ModuleList([nn.LayerNorm(dimensionality) for _ in range(num_layers)])
        self.decoding_head = ConvDecodingHead(dimensionality, patch_size, out_channels)

    def forward(self,
                lidar_input: Tensor | None,
                event_input: Tensor,
                prev_states: dict | None = None,
                crop_positions: Tensor = None) -> tuple[Tensor, Tensor, dict]:  # 新增 crop_positions
        """
        Returns:
            final_depth: The fused dense depth.
            student_depth: The depth predicted purely by events (for Auxiliary Loss).
            new_states: Updated memories.
        """

        # --- 0. Auto Padding ---
        B, C, H, W = event_input.shape
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            event_input = F.pad(event_input, (0, pad_w, 0, pad_h))
            if lidar_input is not None:
                lidar_input = F.pad(lidar_input, (0, pad_w, 0, pad_h))

        H_pad, W_pad = event_input.shape[-2:]
        h_p, w_p = H_pad // self.patch_size, W_pad // self.patch_size

        if lidar_input is None:
            # 创建一个假的 lidar 输入用于形状占位 (如果为None)
            lidar_input = torch.zeros_like(event_input[:, :1])

        # --- Modified: Positional Encoding Logic from DELTA ---

        # 1. 处理 crop_positions
        if crop_positions is None:
            crop_positions = torch.zeros((B, 2), device=event_input.device)
        else:
            # 缩放到 patch 坐标
            crop_positions = crop_positions // self.patch_size

        # 2. 生成网格坐标 (Batch, H_p, W_p, 2)
        positions = torch.empty((B, h_p, w_p, 2), dtype=torch.long, device=event_input.device)

        for b in range(B):
            # 注意：这里的 h_p 和 w_p 是经过 padding 后的特征图尺寸
            pos_x = torch.arange(crop_positions[b, 0], crop_positions[b, 0] + w_p, device=event_input.device)
            pos_y = torch.arange(crop_positions[b, 1], crop_positions[b, 1] + h_p, device=event_input.device)

            # 使用 meshgrid 生成网格
            # indexing='xy' 意味着第一个输出是 x (列), 第二个是 y (行)
            grid_x, grid_y = torch.meshgrid(pos_x, pos_y, indexing='xy')

            # 填入 positions 张量 (注意维度对应: [y, x] 即 [行, 列])
            positions[b, :, :, 0] = grid_x
            positions[b, :, :, 1] = grid_y

        # 3. Reshape 为 (B, L, 2) 并编码
        positions = positions.reshape(B, -1, 2)
        encoded_pos = self.pos_encoder(positions)  # (B, L, D)

        # ------------------------------------------------------

        # --- 1. Init States ---
        if prev_states is None:
            prev_states = {
                'lidar_s': [None] * self.num_layers,
                'event_s': [None] * self.num_layers,
                'densify_s': [None] * self.num_layers,
                'decoder_s': [None] * self.num_layers
            }

        new_states = {
            'lidar_s': [], 'event_s': [], 'densify_s': [], 'decoder_s': []
        }

        # --- 2. Encoding ---
        l_feat, _ = self.lidar_head(lidar_input)
        e_feat, skip_evt_raw = self.event_head(event_input)

        l_feat = l_feat.flatten(2).transpose(1, 2)  # (B, L, D)
        e_feat = e_feat.flatten(2).transpose(1, 2)  # (B, L, D)

        # Modified: Add the generated positional encoding
        l_feat = l_feat + encoded_pos
        e_feat = e_feat + encoded_pos

        # --- 3. Independent Processing ---
        skip_connections = []
        for i in range(self.num_layers):
            # Event Stream (Student)
            e_feat, e_s_new = self.event_ssm_layers[i](e_feat, prev_states['event_s'][i])
            new_states['event_s'].append(e_s_new)

            # LiDAR Stream (Master Content)
            l_feat, l_s_new = self.lidar_ssm_layers[i](l_feat, prev_states['lidar_s'][i])
            new_states['lidar_s'].append(l_s_new)

        # --- 4. [ANGLE 2] Student Prediction (Parallax Learning) ---
        student_depth = self.student_parallax_head(e_feat, h_p, w_p)

        # --- 5. [ANGLE 1] Structure-Guided Densification ---
        densified_feat = l_feat

        for i in range(self.num_layers):
            densified_feat, den_s_new = self.densification_layers[i](
                content=densified_feat,
                structure=e_feat,
                prev_state=prev_states['densify_s'][i]
            )
            new_states['densify_s'].append(den_s_new)
            skip_connections.append(densified_feat + e_feat)

        central_feat = self.post_fusion_mix(densified_feat)

        # --- 6. Decoding ---
        x = central_feat
        for i in range(self.num_layers):
            skip = skip_connections[-(i + 1)]
            x = x + self.skip_norm[i](skip)
            x, d_s_new = self.decoder_ssm_layers[i](x, prev_states['decoder_s'][i])
            new_states['decoder_s'].append(d_s_new)

        # --- 7. Output Head ---
        final_depth = self.decoding_head(x, skip_evt_raw, h_p, w_p)

        # Unpadding
        if pad_h > 0 or pad_w > 0:
            final_depth = final_depth[:, :, :H, :W]
            student_depth = student_depth[:, :, :H, :W]

        return final_depth, student_depth, new_states