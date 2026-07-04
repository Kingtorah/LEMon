#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DELTA-V2: Structure-Aware Dense Depth from Events and LiDAR.
Incorporating Master/Slave guidance and Teacher/Student SSPL learning.
"""

import torch
from torch import nn, Tensor
from torch.nn import functional as F

# Re-using submodules from the original DELTA setup
from models.submodules.shared_submodules import MEHGRU, MultiheadAttentionPreLN, PositionalEncoder2D
from models.submodules.delta_submodules import ConvEncodingHead, ConvDecodingHead


class StructureContentAttention(nn.Module):
    """
  [Innovation 1: Master/Slave]
  Uses Event features to determine the structure (Attention Weights),
  and applies this structure to propagate LiDAR features (Content).

  Q (Structure): Events
  K (Structure): Events
  V (Content):   LiDAR
  """

    def __init__(self, dimensionality: int, nbr_heads: int, ffnn_dimensionality: int):
        super().__init__()
        self.norm_struct = nn.LayerNorm(dimensionality)
        self.norm_content = nn.LayerNorm(dimensionality)

        self.attention = nn.MultiheadAttention(dimensionality, nbr_heads, batch_first=True)

        # Standard FFNN for the content after aggregation
        self.norm_out = nn.LayerNorm(dimensionality)
        self.ffnn = nn.Sequential(
            nn.Linear(dimensionality, ffnn_dimensionality),
            nn.GELU(),
            nn.Linear(ffnn_dimensionality, dimensionality),
        )

    def forward(self, structure_feat: Tensor, content_feat: Tensor) -> Tensor:
        # structure_feat (Events) determines WHERE to look
        # content_feat (LiDAR) determines WHAT to see

        norm_s = self.norm_struct(structure_feat)
        norm_c = self.norm_content(content_feat)

        # Attention(Q=Structure, K=Structure, V=Content)
        # This aggregates LiDAR depths based on Event similarity
        aggregated_content, _ = self.attention(norm_s, norm_s, norm_c, need_weights=False)

        # Residual connection on the CONTENT
        out = content_feat + aggregated_content

        # FFNN
        out = out + self.ffnn(self.norm_out(out))
        return out


class GatedFusionBlock(nn.Module):
    """
  [Hardcore Innovation 3: Uncertainty-Gated Fusion]
  Fuses Events and LiDAR but includes a learnable Gate to handle LiDAR sparsity.
  If LiDAR info is unreliable/missing at a patch, the Gate closes to rely on Event Memory.
  """

    def __init__(self, dimensionality: int, nbr_heads: int, ffnn_dimensionality: int):
        super().__init__()
        self.cross_attn = MultiheadAttentionPreLN(dimensionality, nbr_heads, ffnn_dimensionality)

        # The Gate: Takes [Event_Query, LiDAR_Response] -> outputs scalar weight [0,1] per channel
        self.gate_net = nn.Sequential(
            nn.Linear(dimensionality * 2, dimensionality),
            nn.Sigmoid()
        )

    def forward(self, event_query: Tensor, lidar_kv: Tensor) -> Tensor:
        # Standard Cross Attention: Events query LiDAR
        attn_out = self.cross_attn(event_query, lidar_kv)

        # Calculate uncertainty gate
        # If attn_out is very different from event_query, or if lidar was empty,
        # the network can learn to suppress it.
        gate = self.gate_net(torch.cat([event_query, attn_out], dim=-1))

        # Weighted fusion: Alpha * Attended_LiDAR + (1-Alpha) * Event_History
        # Note: In residual formulation, this acts as a damper on the update
        fused = gate * attn_out  # The residual connection in the main loop handles the "+ Event_History"
        return fused


class SSPLHead(nn.Module):
    """
  [Innovation 2: Teacher/Student SSPL]
  A lightweight head to predict dense depth solely from Event features.
  Used for auxiliary loss against sparse LiDAR.
  """

    def __init__(self, dimensionality: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        # Simple projection from Transformer dim to patch pixels
        self.head = nn.Sequential(
            nn.Linear(dimensionality, 256),
            nn.PReLU(),
            nn.Linear(256, patch_size * patch_size)  # Regress all pixels in patch
        )

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        out = self.head(x)  # (B, N, P*P)

        # Reshape back to image: (B, H, W)
        # Assuming row-major flattening of patches
        h_p = H // self.patch_size
        w_p = W // self.patch_size

        out = out.reshape(B, h_p, w_p, self.patch_size, self.patch_size)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, H, W)
        out = out.unsqueeze(1)  # (B, 1, H, W)
        return out


class LEDepth(nn.Module):
    """
  """

    def __init__(self, lidar_channels: int, event_channels: int, out_channels: int,
                 sa_layers: int, patch_size: int, dimensionality: int, ffnn_dimensionality: int,
                 nbr_heads: int, prop_mem_size: int):
        super(LEDepth, self).__init__()

        self.sa_layers = sa_layers
        self.patch_size = patch_size

        # --- Memories ---
        self.initial_prop_mem = nn.Parameter(torch.empty((prop_mem_size, dimensionality)))
        nn.init.uniform_(self.initial_prop_mem)
        self.saved_lidar = None

        # --- Encoders ---
        self.lidar_head = ConvEncodingHead(patch_size, lidar_channels, dimensionality)
        self.event_head = ConvEncodingHead(patch_size, event_channels, dimensionality)

        # Max resolution assumption (can be adjusted)
        self.pos_encoder = PositionalEncoder2D(dimensionality, (720 // patch_size, 1284 // patch_size))

        # --- Self-Attention Stacks (Intra-modality) ---
        self.lidar_sa = nn.ModuleList(
            [MultiheadAttentionPreLN(dimensionality, nbr_heads, ffnn_dimensionality) for _ in range(sa_layers)])
        self.event_sa = nn.ModuleList(
            [MultiheadAttentionPreLN(dimensionality, nbr_heads, ffnn_dimensionality) for _ in range(sa_layers)])

        # --- [Innovation 1] Structure-Guided Propagation ---
        # Replaces simple cross-attention. Events structure guides LiDAR content propagation.
        self.structure_guided_attn = StructureContentAttention(dimensionality, nbr_heads, ffnn_dimensionality)

        # Standard prop mem update (Events update the memory)
        self.prop_mem_update_ca = MultiheadAttentionPreLN(dimensionality, nbr_heads, ffnn_dimensionality)

        # --- [Hardcore Innovation 3] Gated Central Fusion ---
        # Replaces standard Central CA
        self.gated_fusion_ca = GatedFusionBlock(dimensionality, nbr_heads, ffnn_dimensionality)

        # --- [Innovation 2] SSPL Auxiliary Head ---
        self.aux_sspl_head = SSPLHead(dimensionality, patch_size)

        # --- Memory Update ---
        self.mem_update_gru = MEHGRU(dimensionality, dimensionality)

        # --- Decoder ---
        self.decoder_sa = nn.ModuleList(
            [MultiheadAttentionPreLN(dimensionality, nbr_heads, ffnn_dimensionality) for _ in range(sa_layers)])
        self.skip_norm = nn.ModuleList([nn.LayerNorm(dimensionality) for _ in range(sa_layers)])
        self.decoding_head = ConvDecodingHead(dimensionality, patch_size, out_channels)

    def forward(self, lidar_input: Tensor | None, event_input: Tensor | None, central_mem: Tensor,
                prop_mem: Tensor, crop_positions: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        # --- 0. Data Prep ---
        if lidar_input is None:
            lidar_input = self.saved_lidar.clone()
        else:
            self.saved_lidar = lidar_input.clone()

        batch_size, _, h, w = lidar_input.shape

        # --- 1. Positional Encoding ---
        # (Simplified logic from original code)
        h_p, w_p = h // self.patch_size, w // self.patch_size
        if crop_positions is None:
            crop_positions = torch.zeros((batch_size, 2))
        else:
            crop_positions //= self.patch_size

        positions = torch.empty((batch_size, h_p, w_p, 2), dtype=torch.long).to(lidar_input.device)
        for b in range(batch_size):
            pos_x = torch.arange(crop_positions[b, 0], crop_positions[b, 0] + w_p)
            pos_y = torch.arange(crop_positions[b, 1], crop_positions[b, 1] + h_p)
            grid = torch.meshgrid(pos_x, pos_y, indexing="xy")
            positions[b, :, :, 0] = grid[0]
            positions[b, :, :, 1] = grid[1]

        positions = positions.reshape(batch_size, -1, 2)
        encoded_pos = self.pos_encoder(positions)

        # --- 2. Memory Init ---
        if central_mem is None:
            central_mem = encoded_pos.clone()
        if prop_mem is None:
            prop_mem = self.initial_prop_mem.clone().unsqueeze(0).expand(batch_size, -1, -1)

        # --- 3. Feature Encoding ---
        encod_lidar, _ = self.lidar_head(lidar_input)
        encod_lidar = encod_lidar + encoded_pos

        encod_evt, skip_evt = self.event_head(event_input)
        encod_evt = encod_evt + encoded_pos

        # --- 4. Event Processing (The "Student" & "Structure" Provider) ---
        encod_evts = [encod_evt]
        for sa in self.event_sa:
            encod_evts.append(sa(encod_evts[-1], encod_evts[-1]))

        # [Innovation 2] SSPL: Aux prediction from pure Event features
        # We use the final event features to predict depth
        aux_depth = self.aux_sspl_head(encod_evts[-1], h, w)

        # --- 5. LiDAR Processing (The "Master" Content) ---
        encod_lidars = []

        # [Innovation 1] Structure-Guided Densification
        # Instead of generic cross-attention, we use Event structure to smooth/propagate LiDAR
        # We use the most processed event features to guide the raw-ish LiDAR features
        propagated_lidar = self.structure_guided_attn(structure_feat=encod_evts[-1], content_feat=encod_lidar)
        encod_lidars.append(propagated_lidar)

        for sa in self.lidar_sa:
            encod_lidars.append(sa(encod_lidars[-1], encod_lidars[-1]))

        # --- 6. Prop Memory Update ---
        # Update memory with new Event info
        prop_mem = self.prop_mem_update_ca(prop_mem, encod_evts[0])

        # --- 7. Fusion (The "Hardcore" Gated Fusion) ---
        # [Hardcore Innovation 3] Gated Fusion
        # Events query the LiDAR content, but fusion is gated by confidence
        fused_feat = self.gated_fusion_ca(event_query=encod_evts[-1], lidar_kv=encod_lidars[-1])

        # Add fused info to central memory state via GRU
        central_mem = self.mem_update_gru(fused_feat, central_mem)

        # --- 8. Decoding ---
        pred = central_mem.clone()
        for i in range(self.sa_layers):
            pred = self.decoder_sa[i](pred, pred)
            # Skip connections fusion
            fused_skip = encod_lidars[-i - 2] + encod_evts[-i - 2]
            fused_skip = self.skip_norm[i](fused_skip)
            pred = pred + fused_skip

        final_depth = self.decoding_head(pred, skip_evt, h_p, w_p)

        # Returns: Final Depth, Central Mem, Prop Mem, AND Aux Depth (for SSPL loss)
        return final_depth, central_mem, prop_mem, aux_depth
