#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains custom PyTorch transforms, used by the ALED & DELTA networks.
"""

import torch
from torch import nn, Tensor
from torchvision.transforms.transforms import _setup_size
import torchvision.transforms.functional as F


class PadToMaxSize(nn.Module):
  """
  Padding module, which can take images of varying size as an input, and which pads them to make all
  of them the same predefined size.
  """

  @staticmethod
  def get_params(size: tuple[int, int], output_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """
    From a given input shape and a desired output shape, returns the padding for the left, top,
    bottom, and right sides (in this exact order).
    """

    h, w = size
    th, tw = output_size

    if h > th or w > tw:
      raise ValueError(f"Required padded size {(th, tw)} is smaller than input image size {(h, w)}")

    if w == tw and h == th:
      return 0, 0, 0, 0

    diff_h = th-h
    diff_w = tw-w

    top = diff_h//2
    bottom = diff_h-top
    left = diff_w//2
    right = diff_w-left

    return top, bottom, left, right


  def __init__(self, shape_after_pad: tuple[int, int]):
    super().__init__()
    self.size = tuple(_setup_size(shape_after_pad,
                                  error_msg="Please provide only two dimensions (h, w) for shape_after_pad."))

  def forward(self, x: Tensor) -> Tensor:
    top, bottom, left, right = self.get_params(x.shape[-2:], self.size)
    padded_x = F.pad(x, (left, top, right, bottom))
    return padded_x


class RandomCropAlignedWithPatches(nn.Module):
  """
  Random cropping module, which can optionally make sure that the cropped area is aligned with
  patches (i.e., the cropped area only contains full patches)
  """

  @staticmethod
  def get_params(size: tuple[int, int], output_size: tuple[int, int],
                 patch_size: tuple[int, int] | None) -> tuple[int, int, int, int]:
    """
    From a given input shape, a desired output shape, and an optional patch size, returns the
    y and x coordinates of the top left corner, as well as the cropped height and width.
    """

    h, w = size
    th, tw = output_size

    if patch_size is not None:
      ph, pw = patch_size

    if h < th or w < tw:
      raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

    if w == tw and h == th:
      return 0, 0, h, w

    if patch_size is not None:
      i = torch.randint(0, h//ph - th//ph + 1, size=(1,)).item() * ph
      j = torch.randint(0, w//pw - tw//pw + 1, size=(1,)).item() * pw
    else:
      i = torch.randint(0, h-th+1, size=(1,)).item()
      j = torch.randint(0, w-tw+1, size=(1,)).item()

    return i, j, th, tw

  def __init__(self, crop_size: tuple[int, int], patch_size: tuple[int, int] | None = None):
    super().__init__()
    self.out_size = tuple(_setup_size(crop_size, error_msg="Please provide only two dimensions "
                                                           "(h, w) for crop_size."))
    if patch_size is not None:
      self.patch_size = tuple(_setup_size(patch_size, error_msg="Please provide only two dimensions"
                                                                " (h, w) for patch_size."))
    else:
      self.patch_size = None

  def forward(self, x: Tensor):
    i, j, h, w = self.get_params(x.shape[-2:], self.out_size, self.patch_size)
    cropped_x = x[:, i:i+h, j:j+w]
    return cropped_x
