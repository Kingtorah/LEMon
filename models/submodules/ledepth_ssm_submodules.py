#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains PyTorch submodules, used only by the LEDepth network.
"""

from math import log2

from torch import nn, Tensor

from models.submodules.shared_submodules import ConvexUpsampling


class ConvEncodingHead(nn.Module):
  """
  The encoding head, transforming the input representation to a set of patches, used as part of the
  DELTA network.
  It relies on a succession of convolutions to reach the desired patch size.
  """

  def __init__(self, patch_size: int, in_channels: int, out_dimensionality: int):
    super().__init__()

    # We first check that the given patch size is either a power of 2 (e.g., 16 or 8), or 12
    # If it is a power of 2, then encoding is achieved by consecutive convolutions of stride 2
    # If it is 12, it is achieved by a convolution of stride 3 followed by two convolutions of
    # stride 2
    nbr_convs_patches = log2(patch_size)
    if not nbr_convs_patches.is_integer() and patch_size != 12:
      raise ValueError(f"Patch size {patch_size} is not 12, nor a power of 2!")
    nbr_convs_patches = int(nbr_convs_patches)

    # We set the first convolution, to increase the number of channels without changing the img size
    self.conv_init = nn.Sequential(nn.Conv2d(in_channels,
                                             out_dimensionality//(2**nbr_convs_patches), 5, 1, 2),
                                   nn.PReLU())

    # We set the following convolutions, to reach the desired patch size and dimensionality
    self.conv_patch = nn.ModuleList()
    for i in range(nbr_convs_patches, 0, -1):
      stride = 3 if patch_size == 12 and i == nbr_convs_patches else 2
      self.conv_patch.append(nn.Sequential(nn.Conv2d(out_dimensionality//(2**i),
                                                     out_dimensionality//(2**(i-1)), 5, stride, 2),
                                           nn.PReLU()))

  def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
    """
    The shape of the source input should be (B, C, H, W), where B is the number of batches, C the
    number of channels, and H and W the original height and width.
    The first output is of shape (B, N, D), where B is the number of batches, N the number of
    patches, and D the dimensionality of each patch.
    The second output is an array of the intermediate values, used for the skip connections during
    decoding.
    """

    # We create an array to hold the tensors that will be used for the convex upsampling during
    # decoding
    skip_out = []

    # We begin by applying the first part of the encoding head to reach a shape of (B, D/P, H, W)
    # We also create a copy of the produced Tensor, for the skip connection to the decoder
    out = self.conv_init(x)
    skip_out.append(out)

    # We apply sequentially the downsampling convolutions, to reach a shape of (B, D, H/P, W/P)
    # We also create a copy of the produced Tensor each time, for the skip connection to the decoder
    # (but not for the last one, hence the use of "pop")
    for conv in self.conv_patch:
      out = conv(out)
      skip_out.append(out)
    skip_out.pop()

    # And we finally permute / reshape to reach an output shape of (B, N, D)
    out = out.permute(0, 2, 3, 1)
    out = out.reshape(out.shape[0], -1, out.shape[3])

    return out, skip_out


class ConvDecodingHead(nn.Module):
  """
  The decoding head, transforming the patches back to an image-based representation, used as part of
  the DELTA network.
  This module applies the inverse operations compared to the encoding head, and especially uses
  convex upsampling guided by the skip connections.
  """

  def __init__(self, in_dimensionality: int, patch_size: int, out_channels: int):
    super().__init__()

    # We first check that the given patch size is either a power of 2 (e.g., 16 or 8), or 12
    # If it is a power of 2, then encoding is achieved by consecutive convolutions of stride 2
    # If it is 12, it is achieved by a convolution of stride 3 followed by two convolutions of
    # stride 2
    nbr_ups_patches = log2(patch_size)
    if not nbr_ups_patches.is_integer() and patch_size != 12:
      raise ValueError(f"Patch size {patch_size} is not 12, nor a power of 2!")
    nbr_ups_patches = int(nbr_ups_patches)

    # We set the convex upsampling modules and convolutions, to go back to the original image size
    self.up_conv = nn.ModuleList()
    for i in range(nbr_ups_patches):
      upsample_factor = 3 if patch_size == 12 and i == (nbr_ups_patches-1) else 2
      self.up_conv.append(nn.ModuleList([ConvexUpsampling(in_dimensionality//(2**(i+1)),
                                                          upsample_factor),
                                         nn.Sequential(nn.Conv2d(in_dimensionality//2**i,
                                                                 in_dimensionality//(2**(i+1)), 5,
                                                                 1, 2),
                                                       nn.PReLU())]))

    # We set the final convolution, to reach the desired number of output channels
    self.pred_head = nn.Conv2d(in_dimensionality//(2**nbr_ups_patches), out_channels, 5, 1, 2)

  def forward(self, x: Tensor, skip_evt: list[Tensor], h_p: int, w_p: int):
    """
    The shape of the source input should be (B, N, D), where B is the number of batches, N the
    number of patches, and D the dimensionality.
    The output will be of shape (B, C, H, W), where B is the number of batches, C the number of
    channels of the output, and H and W the original height and width.
    """

    # We reshape the input Tensor to reach a shape of (B, D, H/P, W/P)
    out = x.reshape(x.shape[0], h_p, w_p, x.shape[-1])
    out = out.permute(0, 3, 1, 2)

    # We apply the convex upsampling modules and convolutions to reach a shape of (B, D/P, H, W)
    for i, (upsample, conv) in enumerate(self.up_conv):
      out = upsample(out, skip_evt[-i-1])
      out = conv(out)

    # We apply the last decoding head to reach a shape of (B, C, H, W)
    out = self.pred_head(out)

    return out
