#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains PyTorch submodules, which are used by several networks.
"""

from math import log

import torch
from torch import nn, Tensor


class ResidualBasicEncoder(nn.Module):
  """
  A ResNet Basic encoder, with optional downsampling, used as part of the ALED network.
  It is composed of a two convolutions, each followed by a batch normalization and a PReLU
  activation.
  At the end, the input (potentially downsampled through a 1x1 convolution) is added to the output
  of the last convolution, such that the convolutions only compute the residual.
  Note: an optional instance normalization before the last PReLU can also be enabled, as proposed by
  Pan et al. in their "Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
  article.
  """

  def __init__(self, channels_in: int, channels_out: int, kernel_size: int, stride: int,
               padding: int, use_instance_norm: bool):
    super().__init__()
    self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding)
    self.bn1 = nn.BatchNorm2d(channels_out)
    self.relu1 = nn.PReLU()
    self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size, 1, padding)
    self.bn2 = nn.BatchNorm2d(channels_out)
    self.downsample = stride != 1 or channels_in != channels_out
    if self.downsample:
      self.convd = nn.Conv2d(channels_in, channels_out, 1, stride)
      self.bnd = nn.BatchNorm2d(channels_out)
    self.use_instance_norm = use_instance_norm
    if self.use_instance_norm:
      self.insn = nn.InstanceNorm2d(channels_out)
    self.relu2 = nn.PReLU()

  def forward(self, x: Tensor) -> Tensor:
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample:
      identity = self.convd(identity)
      identity = self.bnd(identity)
    out = out + identity
    if self.use_instance_norm:
      out = self.insn(out)
    out = self.relu2(out)
    return out


class ResidualBottleneckEncoder(nn.Module):
  """
  A ResNet Bottleneck encoder, with optional downsampling, used as part of the ALED network.
  It is composed of a three convolutions, each followed by a batch normalization and a PReLU
  activation.
  At the end, the input (potentially downsampled through a 1x1 convolution) is added to the output
  of the last convolution, such that the convolutions only compute the residual.
  Note: an optional instance normalization before the last PReLU can also be enabled, as proposed by
  Pan et al. in their "Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
  article.
  """

  def __init__(self, channels_in: int, channels_out: int, kernel_size: int, stride: int,
               padding: int, use_instance_norm: bool):
    super().__init__()
    self.conv1 = nn.Conv2d(channels_in, channels_out//4, 1)
    self.bn1 = nn.BatchNorm2d(channels_out//4)
    self.relu1 = nn.PReLU()
    self.conv2 = nn.Conv2d(channels_out//4, channels_out//4, kernel_size, stride, padding)
    self.bn2 = nn.BatchNorm2d(channels_out//4)
    self.relu2 = nn.PReLU()
    self.conv3 = nn.Conv2d(channels_out//4, channels_out, 1)
    self.bn3 = nn.BatchNorm2d(channels_out)
    self.downsample = stride != 1 or channels_in != channels_out
    if self.downsample:
      self.convd = nn.Conv2d(channels_in, channels_out, 1, stride)
      self.bnd = nn.BatchNorm2d(channels_out)
    self.use_instance_norm = use_instance_norm
    if self.use_instance_norm:
      self.insn = nn.InstanceNorm2d(channels_out)
    self.relu3 = nn.PReLU()

  def forward(self, x: Tensor) -> Tensor:
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu2(out)
    out = self.conv3(out)
    out = self.bn3(out)
    if self.downsample:
      identity = self.convd(identity)
      identity = self.bnd(identity)
    out = out + identity
    if self.use_instance_norm:
      out = self.insn(out)
    out = self.relu3(out)
    return out


class ConvGRU(nn.Module):
  """
  The Convolutional Gated Recurrent Unit (ConvGRU) submodule, used as part of the ALED network.
  Adapted from https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
  """

  def __init__(self, input_size: int, hidden_size: int, kernel_size: int):
    super().__init__()
    padding = kernel_size // 2
    self.hidden_size = hidden_size
    self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
    self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
    self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

    nn.init.orthogonal_(self.reset_gate.weight)
    nn.init.orthogonal_(self.update_gate.weight)
    nn.init.orthogonal_(self.out_gate.weight)
    nn.init.constant_(self.reset_gate.bias, 0.)
    nn.init.constant_(self.update_gate.bias, 0.)
    nn.init.constant_(self.out_gate.bias, 0.)

  def forward(self, x: Tensor, prev_state: Tensor | None) -> Tensor:
    # Generate empty prev_state if None is provided
    if prev_state is None:
      batch_size = x.data.size()[0]
      spatial_size = x.data.size()[2:]
      state_size = [batch_size, self.hidden_size] + list(spatial_size)
      prev_state = torch.zeros(state_size, dtype=x.dtype, device=x.device)

    # Data size is (B, C, H, W)
    stacked_inputs = torch.cat([x, prev_state], dim=1)
    update = torch.sigmoid(self.update_gate(stacked_inputs))
    reset = torch.sigmoid(self.reset_gate(stacked_inputs))
    out_inputs = torch.tanh(self.out_gate(torch.cat([x, prev_state * reset], dim=1)))
    new_state = prev_state * (1 - update) + out_inputs * update

    return new_state


class MEHGRU(nn.Module):
  """
  The Gated Recurrent Unit (GRU) submodule, used as part of the DELTA network.
  We use here an adapted implementation (similar to the convGRU of ALED), to be able to have a
  hidden state with multiple elements.
  """

  def __init__(self, input_dim: int, hidden_dim: int):
    super().__init__()

    # The reset, update, and output gates
    self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
    self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
    self.out_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

  def forward(self, x: Tensor, prev_state: Tensor) -> Tensor:
    """
    Inputs should be of size (B, N, D_in) for x, and (B, N, D_h) for prev_state, where B is the
    batch size, N is the number of elements, D_in the dimensionality of the input, and D_h the
    dimensionality of the hidden state.
    """

    # We begin by concatenating the inputs, along the dimensionality dimension
    # The shape is (B, N, D_in+D_h)
    stacked_inputs = torch.cat([x, prev_state], dim=2)

    # We compute the update and reset vectors
    # Their shape is (B, N, D_h)
    update = torch.sigmoid(self.update_gate(stacked_inputs))
    reset = torch.sigmoid(self.reset_gate(stacked_inputs))

    # We compute the updated values based on the input and the reseted previous state
    # The shape is (B, N, D_h)
    out_inputs = torch.tanh(self.out_gate(torch.cat([x, prev_state * reset], dim=2)))

    # We finish by computing the new state, as a weighted sum between the previous state and the
    # updated values
    # The shape of the new state is (B, N, D_h)
    new_state = prev_state * (1 - update) + out_inputs * update

    # And we return it
    return new_state


class ConvexUpsampling(nn.Module):
  """
  The convex upsampling submodule, used as part of ALED & DELTA networks.
  It is a learnt alternative to bilinear upsampling, originally described in the "RAFT: Recurrent
  All-Pairs Field Transforms for Optical Flow" article by Z. Teed and J. Deng.
  """

  def __init__(self, channels_in_guide: int, upsample_factor: int):
    super().__init__()
    self.upsample_factor = upsample_factor
    self.mask_conv0 = nn.Conv2d((upsample_factor**2)*channels_in_guide, 256, 3, padding=1)
    self.mask_relu = nn.PReLU()
    self.mask_conv1 = nn.Conv2d(256, self.upsample_factor**2 * 9, 1)
    self.unfold = nn.Unfold((3, 3), padding=1)
    # self.mask_factor = nn.Buffer(torch.tensor([1.0]))
    self.register_buffer('mask_factor', torch.tensor([1.0]))

  def forward(self, x: Tensor, guide: Tensor) -> Tensor:
    # Guide folding
    batch_size_guide, channels_guide, height_guide, width_guide = guide.shape
    mask = torch.empty(batch_size_guide, (self.upsample_factor**2)*channels_guide,
                       height_guide//self.upsample_factor, width_guide//self.upsample_factor,
                       device=guide.device)
    for c in range(self.upsample_factor**2):
      mask[:, c*channels_guide:(c+1)*channels_guide, :, :] = guide[:, :, c//self.upsample_factor::self.upsample_factor, c%self.upsample_factor::self.upsample_factor]

    # Mask computation
    # Note that the mask_factor is kept for backwards compatibility with old ALED/DELTA checkpoints,
    # which were using a factor of 0.25 (which was originally part of RAFT's code, for more details
    # please see https://github.com/princeton-vl/RAFT/issues/24 and
    # https://github.com/princeton-vl/RAFT/issues/119)
    mask = self.mask_conv0(mask)
    mask = self.mask_relu(mask)
    mask = self.mask_conv1(mask)
    mask = self.mask_factor.item() * mask

    # Mask reshaping and activation function
    batch_size, channels, height, width = x.shape
    mask = mask.view(batch_size, 1, 9, self.upsample_factor, self.upsample_factor, height, width)
    mask = torch.softmax(mask, dim=2)

    # Upsampling
    x_up = self.unfold(x)
    x_up = x_up.view(batch_size, channels, 9, 1, 1, height, width)
    x_up = torch.sum(mask * x_up, dim=2)
    x_up = x_up.permute(0, 1, 4, 2, 5, 3)
    x_up = x_up.reshape(batch_size, channels, self.upsample_factor*height, self.upsample_factor*width)
    return x_up


class PositionalEncoder2D(nn.Module):
  """
  The sinusoidal position encoder, generalized to 2-dimensional images, used as part of the DELTA
  network.
  This code is inspired by the one of LoFTR (https://github.com/zju3dv/LoFTR).
  """

  def __init__(self, dimensionality: int, max_shape: tuple[int, int]):
    super().__init__()

    # We compute the x and y positions as tensors of shape (1, H, W)
    y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
    x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)

    # We compute the division term of shape D//4, and then cast it to shape (D//4, 1, 1)
    div_term = torch.exp(torch.arange(0, dimensionality//2, 2).float() *
                         (-log(10000.0) / (dimensionality//2)))
    div_term = div_term[:, None, None]

    # We initialize the position encoding as a tensor of shape (D, H, W)
    pe = torch.zeros((dimensionality, *max_shape))
    pe[0::4, :, :] = torch.sin(x_position * div_term)
    pe[1::4, :, :] = torch.cos(x_position * div_term)
    pe[2::4, :, :] = torch.sin(y_position * div_term)
    pe[3::4, :, :] = torch.cos(y_position * div_term)

    # We save it as a buffer
    self.register_buffer("pe", pe, persistent=False)

  def forward(self, pos: Tensor) -> Tensor:
    """
    `pos` should be a Tensor of positions of shape (B, N, 2), where 2 is the (x, y) position
    """

    # We extract the encoded positions (shape (D, B, N))
    encoded_pos = self.pe[:, pos[:, :, 1], pos[:, :, 0]]

    # We reshape them to (B, N, D)
    encoded_pos = encoded_pos.permute(1, 2, 0)

    # We return the encoded positions
    return encoded_pos


# class MultiheadAttentionPreLN(nn.Module):
#   """
#   The multihead attention module, used as part of the DELTA network.
#   This module can be used to either compute self- or cross-attention (but it is only used here for
#   self-attention).
#   We follow the "Pre-LN Transformer" format; see the "On Layer Normalization in the Transformer
#   Architecture" paper by Xiong et al. (ICML 2020, https://arxiv.org/pdf/2002.04745.pdf) for more
#   details (much more stable / better gradients repartition)
#   """
#
#   def __init__(self, dimensionality: int, nbr_heads: int, ffnn_dimensionality: int):
#     super().__init__()
#
#     # The layer normalization before computing the attention
#     self.norm1 = nn.LayerNorm(dimensionality)
#
#     # The attention activation
#     self.attention = nn.MultiheadAttention(dimensionality, nbr_heads, batch_first=True)
#
#     # The layer normalization before the feed-forward NN
#     self.norm2 = nn.LayerNorm(dimensionality)
#
#     # The feed-forward neural network (with a GELU activation function)
#     self.ffnn = nn.Sequential(
#       nn.Linear(dimensionality, ffnn_dimensionality),
#       nn.GELU(),
#       nn.Linear(ffnn_dimensionality, dimensionality),
#     )
#
#   def forward(self, queries: Tensor, keys_values: Tensor) -> Tensor:
#     """
#     The shape of the input queries and keys/values should be (B, N, D), where B is the number of
#     batches, N is the number of elements, and D is the dimensionality.
#     """
#
#     # We begin by normalizing the inputs, and we apply the attention on them
#     normed_queries = self.norm1(queries)
#     normed_keys_values = self.norm1(keys_values)
#     merged_att_values, _ = self.attention(normed_queries, normed_keys_values, normed_keys_values,
#                                           need_weights=False)
#
#     # We sum the input and the attended values
#     merged_att_values_res = queries + merged_att_values
#
#     # We normalize this tensor, and we apply the FFNN on it
#     ffnn_input = self.norm2(merged_att_values_res)
#     ffnn_output = self.ffnn(ffnn_input)
#
#     # We sum the attended input and the ffnn output
#     out = merged_att_values_res + ffnn_output
#
#     # We finally return the output
#     return out
class MultiheadAttentionPreLN(nn.Module):
  """
  The multihead attention module, used as part of the DELTA network.
  This module can be used to either compute self- or cross-attention.
  We follow the "Pre-LN Transformer" format.
  """

  def __init__(self, dimensionality: int, nbr_heads: int, ffnn_dimensionality: int):
    super().__init__()

    # The layer normalization before computing the attention
    self.norm1 = nn.LayerNorm(dimensionality)

    # The attention activation
    self.attention = nn.MultiheadAttention(dimensionality, nbr_heads, batch_first=True)

    # The layer normalization before the feed-forward NN
    self.norm2 = nn.LayerNorm(dimensionality)

    # The feed-forward neural network (with a GELU activation function)
    self.ffnn = nn.Sequential(
      nn.Linear(dimensionality, ffnn_dimensionality),
      nn.GELU(),
      nn.Linear(ffnn_dimensionality, dimensionality),
    )

  def forward(self, queries: Tensor, keys_values: Tensor) -> Tensor:
    """
    The shape of the input queries and keys/values should be (B, N, D).
    """

    # We begin by normalizing the inputs
    normed_queries = self.norm1(queries)
    normed_keys_values = self.norm1(keys_values)

    # =========================================================================
    # 🔴 FIX: 强制使用 FP32 计算 Attention，防止 FP16 溢出导致 NaN
    # =========================================================================
    with torch.autocast("cuda", enabled=False):
      # 1. 临时将输入转为 float32 (全精度)
      # 注意：nn.MultiheadAttention 的权重通常已经是 FP32，所以输入也必须是 FP32
      q_fp32 = normed_queries.float()
      k_v_fp32 = normed_keys_values.float()

      # 2. 运行 Attention (此时绝对安全，不会溢出)
      merged_att_values, _ = self.attention(q_fp32, k_v_fp32, k_v_fp32,
                                            need_weights=False)

    # 3. 将结果转回原来的数据类型 (如 FP16)，以便后续网络继续节省显存
    merged_att_values = merged_att_values.to(dtype=queries.dtype)
    # =========================================================================

    # We sum the input and the attended values
    merged_att_values_res = queries + merged_att_values

    # We normalize this tensor, and we apply the FFNN on it
    ffnn_input = self.norm2(merged_att_values_res)
    ffnn_output = self.ffnn(ffnn_input)

    # We sum the attended input and the ffnn output
    out = merged_att_values_res + ffnn_output

    # We finally return the output
    return out