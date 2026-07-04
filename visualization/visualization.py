#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains visualization functions, which can be used to convert raw tensors of data (event
volumes, raw depth images, ...) to a humanly understandble image, which can then be saved or
displayed for overview/debug/... purposes.
"""

from math import log

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def lidar_proj_to_img(lidar_proj: Tensor, bigger_points: bool = False) -> Tensor:
  """
  Converts a LiDAR projection Tensor of shape [B, 1, H, W] into a visualizable Tensor of shape
  [B, 3, H, W], where each LiDAR point is colored based on its log depth, and the background is set
  to gray for better visibility.
  If required, the size of the projected LiDAR points can be increased for a better visibility (but
  at the cost of more computation)
  """

  color_map = plt.get_cmap("inferno")
  lidar_proj_log = lidar_proj.clone()
  lidar_proj_log.squeeze_(dim=1)
  if bigger_points:
    lidar_proj_log[lidar_proj_log == 0] = 1000
    lidar_proj_log_orig = lidar_proj_log.clone()
    lidar_proj_log[:, :-1, :] = torch.min(lidar_proj_log_orig[:, 1:, :], lidar_proj_log[:, :-1, :])
    lidar_proj_log[:, 1:, :] = torch.min(lidar_proj_log_orig[:, :-1, :], lidar_proj_log[:, 1:, :])
    lidar_proj_log[:, :, :-1] = torch.min(lidar_proj_log_orig[:, :, 1:], lidar_proj_log[:, :, :-1])
    lidar_proj_log[:, :, 1:] = torch.min(lidar_proj_log_orig[:, :, :-1], lidar_proj_log[:, :, 1:])
    lidar_proj_log[lidar_proj_log == 1000] = 0
  lidar_proj_log = torch.log(lidar_proj_log+1) / log(2)
  lidar_proj_img_np = color_map(lidar_proj_log.cpu())
  lidar_proj_img_np[lidar_proj_img_np == color_map(0)] = 0.0 #black
  lidar_proj_img = torch.from_numpy(lidar_proj_img_np).permute((0, 3, 1, 2))
  lidar_proj_img_no_alpha = lidar_proj_img[:, :3, :, :]
  return lidar_proj_img_no_alpha


def event_volume_to_img(event_volume: Tensor) -> Tensor:
  """
  Converts an event volume Tensor of shape [B, C, H, W] into a visualizable Tensor of shape
  [B, 3, H, W], where the C temporal bins are squashed, and the negative events are displayed in
  blue while the positive ones are in red, with the background set to gray for better visibility
  """

  batches, _, height, width = event_volume.shape
  event_volume_binary_neg = event_volume < 0
  event_volume_binary_pos = event_volume > 0
  event_volume_img = torch.zeros((batches, 3, height, width))
  event_volume_img[:, 2, :, :] = torch.sum(event_volume_binary_neg, dim=1)
  event_volume_img[:, 0, :, :] = torch.sum(event_volume_binary_pos, dim=1)
  gray_mask = torch.sum(event_volume_img, dim=1, keepdim=True) == 0
  gray_mask = gray_mask.expand((-1, 3, -1, -1))
  # event_volume_img[gray_mask] = 100/255
  event_volume_img[gray_mask] = 0.0 #black
  return event_volume_img


def depth_image_to_img(depth_image: Tensor) -> Tensor:
  """
  Converts a depth image Tensor of shape [B, 1, H, W] into a visualizable Tensor of shape
  [B, 3, H, W], where each pixel is colored based on its log depth (while ensuring that the values
  are in the range [0, 1]), and where pixels with no value are colored in gray for better visibility
  """

  # color_map = plt.get_cmap("turbo")
  color_map = plt.get_cmap("inferno")
  nan_mask = torch.isnan(depth_image)
  nan_mask = nan_mask.expand((-1, 3, -1, -1))
  depth_image_log = depth_image.clone()
  depth_image_log.squeeze_(dim=1)
  depth_image_log[depth_image_log < 0.0] = 0.0
  depth_image_log[depth_image_log > 1.0] = 1.0
  depth_image_log = torch.log(depth_image_log+1) / log(2)
  depth_image_img_np = color_map(depth_image_log.cpu())
  depth_image_img = torch.from_numpy(depth_image_img_np).permute((0, 3, 1, 2))
  depth_image_img_no_alpha = depth_image_img[:, :3, :, :]
  # depth_image_img_no_alpha[nan_mask] = 100/255
  depth_image_img_no_alpha[nan_mask] = 0.0 #black
  return depth_image_img_no_alpha

