#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains PyTorch submodules, used only by the ALED network.
"""

from torch import nn, Tensor

from models.submodules.shared_submodules import ConvexUpsampling


class ConvEncodingHead(nn.Module):
  """
  The encoding head, transforming the input representation to a tensor of fixed size, used as part
  of the ALED network.
  It is composed of a single convolution, followed by a PReLU activation function.
  """

  def __init__(self, channels_in: int, channels_out: int, kernel_size: int, stride: int,
               padding: int):
    super().__init__()
    self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding)
    self.relu = nn.PReLU()

  def forward(self, x: Tensor) -> Tensor:
    out = self.conv(x)
    out = self.relu(out)
    return out


class Decoder(nn.Module):
  """
  The decoder submodule, used as part of the ALED network.
  It is composed of a convex upsampling, followed by a convolution and a PReLU activation.
  """

  def __init__(self, channels_in: int, channels_in_guide: int, channels_out: int,
               upsample_factor: int, kernel_size: int, stride: int, padding: int):
    super().__init__()
    self.upsample = ConvexUpsampling(channels_in_guide, upsample_factor)
    self.upsample_factor = upsample_factor
    self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding)
    self.relu = nn.PReLU()

  def forward(self, x: Tensor, guide: Tensor) -> Tensor:
    if self.upsample_factor > 1:
      out = self.upsample(x, guide)
    else:
      out = x
    out = self.conv(out)
    out = self.relu(out)
    return out
