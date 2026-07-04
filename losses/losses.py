#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains PyTorch losses, which can be used for training the ALED & DELTA networks.
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class L1MSGLoss(nn.Module):
  """
  The traditional L1 loss, with the addition of a multi-scale gradient matching term, as used in the
  "3D Ken Burns Effect from a Single Image" article by Niklaus et al.
  Compared to the reference article, the scale invariant term is not used here, as we are working on
  scaled data.
  """

  def __init__(self, scales: int):
    super().__init__()
    self.scales = scales

  def forward(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """
    The loss computation function.
    The L1 and MSG losses are returned separately, so that different weights can be applied to them.
    """

    # The L1 loss is firstly computed
    not_nan_mask = ~torch.isnan(target)
    l1_loss = F.l1_loss(pred[not_nan_mask], target[not_nan_mask])

    # We then compute the multiscale gradient matching loss
    all_scale_invar_grad_pred = []
    all_scale_invar_grad_target = []

    for scale in range(self.scales):
      shift_px = 2**scale
      shifted_pred_x = pred.clone()
      shifted_pred_y = pred.clone()
      shifted_target_x = target.clone()
      shifted_target_y = target.clone()
      shifted_pred_x[:, :, :, :-shift_px] = pred[:, :, :, shift_px:]
      shifted_pred_y[:, :, :-shift_px, :] = pred[:, :, shift_px:, :]
      shifted_target_x[:, :, :, :-shift_px] = target[:, :, :, shift_px:]
      shifted_target_y[:, :, :-shift_px, :] = target[:, :, shift_px:, :]
      scale_invar_grad_pred_x = shifted_pred_x-pred
      scale_invar_grad_pred_y = shifted_pred_y-pred
      scale_invar_grad_target_x = shifted_target_x-target
      scale_invar_grad_target_y = shifted_target_y-target

      nan_mask_x = torch.isnan(scale_invar_grad_target_x)
      nan_mask_y = torch.isnan(scale_invar_grad_target_y)
      scale_invar_grad_pred_x[nan_mask_x] = 0
      scale_invar_grad_pred_y[nan_mask_y] = 0
      scale_invar_grad_target_x[nan_mask_x] = 0
      scale_invar_grad_target_y[nan_mask_y] = 0

      all_scale_invar_grad_pred.append(scale_invar_grad_pred_x)
      all_scale_invar_grad_pred.append(scale_invar_grad_pred_y)
      all_scale_invar_grad_target.append(scale_invar_grad_target_x)
      all_scale_invar_grad_target.append(scale_invar_grad_target_y)

    all_scale_invar_grad_pred_concat = torch.concat(all_scale_invar_grad_pred, dim=1)
    all_scale_invar_grad_target_concat = torch.concat(all_scale_invar_grad_target, dim=1)

    diff = all_scale_invar_grad_pred_concat - all_scale_invar_grad_target_concat
    if torch.isinf(diff ** 2).any():
      print("!!! Loss computation: Squared difference overflowed in FP16 !!!")

    ms_grad_match_loss = torch.mean(torch.sum(diff**2, dim=1)).float()

    # Finally, we return the L1 loss and the multiscale gradient matching loss
    return l1_loss, ms_grad_match_loss
