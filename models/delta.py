#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains the PyTorch code for the DELTA network, as originally described in the "DELTA:
Dense Depth from Events and LiDAR using Transformer's Attention" article (CVPRW 2025).
"""

import torch
from torch import nn, Tensor
# import sys
# import os
#
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
#
# # 2. 把这两个路径强行塞进 Python 的环境变量里
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)
#
# # 3. 裸奔导包！不要加 try...except。行就行，不行让它立刻报出真实的 ImportError
# from submodules.shared_submodules import MEHGRU, MultiheadAttentionPreLN, PositionalEncoder2D
# from submodules.delta_submodules import ConvEncodingHead, ConvDecodingHead

from models.submodules.shared_submodules import MEHGRU, MultiheadAttentionPreLN, PositionalEncoder2D
from models.submodules.delta_submodules import ConvEncodingHead, ConvDecodingHead


class DELTA(nn.Module):
  """
  The DELTA network, as described in the article.
  It is composed of two branches, one for the events, the other one for the LiDAR scans, and
  uses self- and cross-attention for encoding/decoding and fusion, and GRUs for memory purposes.
  """

  def __init__(self, lidar_channels: int, event_channels: int, out_channels: int,
               sa_layers: int, patch_size: int, dimensionality: int, ffnn_dimensionality: int,
               nbr_heads: int, prop_mem_size: int):
    super(DELTA, self).__init__()

    # We save the number of self-attention layers and the patch size
    self.sa_layers = sa_layers
    self.patch_size = patch_size

    # The initial state of the propagation memory (which is learnt)
    self.initial_prop_mem = nn.Parameter(torch.empty((prop_mem_size, dimensionality)))
    nn.init.uniform_(self.initial_prop_mem)

    # The saved LiDAR data, used when there is no new LiDAR data available
    self.saved_lidar = None

    # The encoding heads for the LiDAR and event inputs
    self.lidar_head = ConvEncodingHead(patch_size, lidar_channels, dimensionality)
    self.event_head = ConvEncodingHead(patch_size, event_channels, dimensionality)

    # The 2D positional encoder
    # We consider here that the max input resolution for our network is 1284x720, but this can be
    # changed if needed
    self.pos_encoder = PositionalEncoder2D(dimensionality, (720//patch_size, 1284//patch_size))

    # The N self-attention modules for the encoded LiDAR and events data
    self.lidar_sa = nn.ModuleList([MultiheadAttentionPreLN(dimensionality, nbr_heads, ffnn_dimensionality) for _ in range(sa_layers)])
    self.event_sa = nn.ModuleList([MultiheadAttentionPreLN(dimensionality, nbr_heads, ffnn_dimensionality) for _ in range(sa_layers)])

    # The cross-attention modules to update the prop. memory and to use it to propagate the LiDAR
    self.prop_mem_update_ca = MultiheadAttentionPreLN(dimensionality, nbr_heads, ffnn_dimensionality)
    self.lidar_prop_mem_ca = MultiheadAttentionPreLN(dimensionality, nbr_heads, ffnn_dimensionality)

    # The central cross-attention between the propagated LiDAR and the events
    self.central_ca = MultiheadAttentionPreLN(dimensionality, nbr_heads, ffnn_dimensionality)

    # The GRU-based memory update
    self.mem_update_gru = MEHGRU(dimensionality, dimensionality)

    # The self-attention modules for the decoder
    self.decoder_sa = nn.ModuleList([MultiheadAttentionPreLN(dimensionality, nbr_heads, ffnn_dimensionality) for _ in range(sa_layers)])

    # The layer normalization modules for the skip connections
    self.skip_norm = nn.ModuleList([nn.LayerNorm(dimensionality) for _ in range(sa_layers)])

    # The final feed-forward decoder
    self.decoding_head = ConvDecodingHead(dimensionality, patch_size, out_channels)


  def forward(self, lidar_input: Tensor | None, event_input: Tensor | None, central_mem: Tensor,
              prop_mem: Tensor, crop_positions: Tensor = None) -> tuple[Tensor, Tensor, Tensor]:
    # PART 0: CHECKING IF LIDAR DATA IS AVAILABLE
    # If not, we initialize it from the saved one
    # Otherwise, we replace the saved LiDAR by the new one
    if lidar_input is None:
      lidar_input = self.saved_lidar.clone()
    else:
      self.saved_lidar = lidar_input.clone()

    # PART 1: GETTING THE POSITIONAL ENCODING
    # crop_positions is of shape (B, 2), and contains the position of the top-left patch
    # We want the positions of all patches, of shape (B, L, 2)

    # We begin by extracting the batch size, and the patched height and width
    batch_size, _, h, w = lidar_input.shape
    h_p = h // self.patch_size
    w_p = w // self.patch_size

    # If crop_positions is None (i.e., no cropping), we set it to (0, 0) for every element in the
    # batch (i.e., the top-left patch is the one at position (0, 0))
    # Otherwise, we scale it with respect to the patch size
    if crop_positions is None:
      crop_positions = torch.zeros((batch_size, 2))
    else:
      crop_positions //= self.patch_size

    # We create an empty Tensor which will hold the positions
    positions = torch.empty((batch_size, h_p, w_p, 2), dtype=torch.long).to(lidar_input.device)

    # Then, for each batch, we fill this Tensor accordingly
    for b in range(batch_size):
      pos_x = torch.arange(crop_positions[b, 0], crop_positions[b, 0]+w_p)
      pos_y = torch.arange(crop_positions[b, 1], crop_positions[b, 1]+h_p)
      positions_meshgrid = torch.meshgrid(pos_x, pos_y, indexing="xy")
      positions[b, :, :, 0] = positions_meshgrid[0]
      positions[b, :, :, 1] = positions_meshgrid[1]

    # We reshape the positions, to have a correct shape of (B, L, 2)
    positions = positions.reshape(batch_size, -1, 2)

    # And we finish by encoding them through the 2D positional encoder
    encoded_pos = self.pos_encoder(positions)

    # PART 2: VERIFYING IF THE GIVEN MEMORIES ARE INITIALIZED, OTHERWISE INITIALIZE THEM
    # We initialize the central memory if necessary, using the positional encoding
    if central_mem is None:
      central_mem = encoded_pos.clone()

    # We initialize the propagation memory if necessary
    if prop_mem is None:
      # We get the learnt initial memory, and duplicate it for every batch
      # For more details, see:
      # https://discuss.pytorch.org/t/learn-initial-hidden-state-h0-for-rnn/10013/
      # https://github.com/AlbertoSabater/EventTransformer/blob/main/models/EvT.py#L258
      prop_mem = self.initial_prop_mem.clone().unsqueeze(0).expand(batch_size, -1, -1)

    # PART 3: ENCODING THE LIDAR DATA
    # Since we'll use the output of each SA layer for skip connections, we save them in a list
    encod_lidars = []

    # We first go through the LiDAR encoding head
    encod_lidar, _ = self.lidar_head(lidar_input)

    # We add the positional embedding
    encod_lidar = encod_lidar + encoded_pos

    # We use the prop. memory to propagate the LiDAR data
    propagated_lidar = self.lidar_prop_mem_ca(encod_lidar, prop_mem)
    encod_lidars.append(propagated_lidar)

    # And we apply the self-attention
    for i in range(self.sa_layers):
      encod_lidars.append(self.lidar_sa[i](encod_lidars[-1], encod_lidars[-1]))

    # PART 4: ENCODING THE EVENTS DATA
    # Since we'll use the output of each SA layer for skip connections, we save them in a list
    encod_evts = []

    # We first go through the events encoding head
    encod_evt, skip_evt = self.event_head(event_input)

    # We add the positional embedding
    encod_evt = encod_evt + encoded_pos

    # And we use the it as the input for the self-attention encoder
    encod_evts.append(encod_evt)

    # We apply the self-attention
    for i in range(self.sa_layers):
      encod_evts.append(self.event_sa[i](encod_evts[-1], encod_evts[-1]))

    # PART 5: UPDATING THE PROPAGATION MEMORY WITH THE EVENTS FOR THE NEXT TIME
    # We update the prop. memory with the encoded events (encoded only by the head, not after SA)
    prop_mem = self.prop_mem_update_ca(prop_mem, encod_evts[0])

    # PART 6: USING THE PROPAGATED LIDAR AND THE EVENTS TO UPDATE THE CENTRAL MEMORY
    # We apply the central CA between the LiDAR and the events
    fused_lidar_evts = self.central_ca(encod_evts[-1], encod_lidars[-1])

    # We compute the new memory by using the GRU module
    central_mem = self.mem_update_gru(fused_lidar_evts, central_mem)

    # PART 7: DECODING THE MEMORY TO PREDICT THE FINAL DEPTH VALUES
    # Our initial prediction is based on this memory
    pred = central_mem.clone()

    # We apply the self-attention layers on it, and add the summed and normalized inputs through the
    # skip connections after every layer
    for i in range(self.sa_layers):
      pred = self.decoder_sa[i](pred, pred)
      fused_skip = encod_lidars[-i-2] + encod_evts[-i-2]
      fused_skip = self.skip_norm[i](fused_skip)
      pred = pred + fused_skip

    # We apply the final decoding head
    h_p = lidar_input.shape[2] // self.patch_size
    w_p = lidar_input.shape[3] // self.patch_size
    pred = self.decoding_head(pred, skip_evt, h_p, w_p)

    # We return the prediction, the central memory, and the updated propagation memory
    return pred, central_mem, prop_mem


import time


# =========================================================================
# 测试脚本 (Main)
# =========================================================================

import torch





def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"🚀 Running DELTA on {device}...")

  # 1. 配置模型参数
  # 这里设置了一些合理的 Transformer/CNN 默认超参数，你可以根据实际论文配置进行修改
  model = DELTA(1, 4, 1, 2, 16, 1024, 4096, 4, 128).to(device)

  # 2. 模拟数据 (使用与上文相同的 720x1280 分辨率)
  h, w = 720, 1280
  # h, w = 264, 348
  x_lidar = torch.randn(1, 1, h, w).to(device)
  x_event = torch.randn(1, 4, h, w).to(device)

  # 初始状态下没有记忆，设为 None
  central_mem = None
  prop_mem = None

  print(f"📦 Input: LiDAR {x_lidar.shape}, Event {x_event.shape}")

  # ==========================================
  # A. 计算参数量 (Parameters)
  # ==========================================
  total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"📊 模型参数量 (Trainable Params): {total_params / 1e6:.2f} M")

  # ==========================================
  # B. 计算计算量 (FLOPs / MACs)
  # ==========================================
  try:
    from thop import profile
    print("🧮 正在计算 FLOPs (可能会花费几秒钟)...")
    # 注意：thop 传入的 inputs 必须对应 forward 函数的参数顺序
    macs, params = profile(model, inputs=(x_lidar, x_event, central_mem, prop_mem), verbose=False)
    print(f"🧮 计算量 (MACs): {macs / 1e9:.2f} G")
    print(f"🧮 预估 FLOPs: {(macs * 2) / 1e9:.2f} G")
  except ImportError:
    print("【提示】未检测到 thop 库，跳过 FLOPs 计算。建议运行 `pip install thop` 后重试。")

  # ==========================================
  # C. 计算单次运行时间 (Inference Time)
  # ==========================================
  try:
    print("⏱️ 开始测速 (包含 Warm-up)...")
    model.eval()
    with torch.no_grad():
      # 预热 (Warm-up): 循环传递记忆以模拟真实推理流
      c_mem, p_mem = None, None
      for _ in range(5):
        _, c_mem, p_mem = model(x_lidar, x_event, c_mem, p_mem)

      # 严格同步 GPU 开始时间
      if device == "cuda":
        torch.cuda.synchronize()
      start_time = time.perf_counter()

      # 实际测速运行 (继续使用上一帧的记忆)
      pred, c_mem, p_mem = model(x_lidar, x_event, c_mem, p_mem)

      # 严格同步 GPU 结束时间
      if device == "cuda":
        torch.cuda.synchronize()
      end_time = time.perf_counter()

    infer_time_ms = (end_time - start_time) * 1000
    print(f"⚡ 单次前向传播时间: {infer_time_ms:.2f} ms (约 {1000 / infer_time_ms:.1f} FPS)")

    # 验证输出
    print(f"✅ Success! Depth Output Shape: {pred.shape}")

    # 验证分辨率是否匹配 (有些解码器可能会因为 patch size 的除法导致边缘像素丢失)
    if pred.shape[-2:] == (h, w):
      print("🎉 Resolution Match!")
    else:
      print(f"⚠️ Resolution Mismatch: Expected {(h, w)}, Got {pred.shape[-2:]}")

  except Exception as e:
    print(f"❌ Error during execution: {e}")
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
  measure_memory_cost(model, (x_lidar, x_event, c_mem, p_mem))


if __name__ == "__main__":
  main()