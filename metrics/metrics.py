#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains several metrics, used to evaluate the ALED & DELTA networks.
"""

import torch
from torch import nn, Tensor


def l1_error(pred_depth_map: Tensor, gt_depth_map: Tensor, reduction: str = "mean") -> float:
  """
  The basic L1 error, which is the average absolute error over the whole depth map.
  Both inputs should have the same shape, but without any constraint on a specific shape to use.
  Inputs should also *not* contain any NaN value.
  """

  # We create the criterion
  l1_criterion = nn.L1Loss(reduction=reduction)

  # We apply it on the inputs, to compute the error
  l1_err = l1_criterion(pred_depth_map, gt_depth_map).item()

  # And we return it
  return l1_err


def absrel_error(pred_depth_map: Tensor, gt_depth_map: Tensor, reduction: str = "mean") -> float:
  """
  The Absolute Relative (AbsRel) error, which is the average absolute error over the whole depth
  map, normalized with respect to the ground truth depths.
  Both inputs should have the same shape, but without any constraint on a specific shape to use.
  Inputs should also *not* contain any NaN value.
  """

  # We initialize the reduction function
  if reduction == "mean":
    reduction_fn = torch.mean
  elif reduction == "sum":
    reduction_fn = torch.sum
  else:
    raise ValueError(f"Unknown reduction '{reduction}'")

  # We compute the error on the inputs
  absrel_err = reduction_fn(torch.abs(pred_depth_map-gt_depth_map)/gt_depth_map).item()

  # And we return it
  return absrel_err


def delta_error(pred_depth_map: Tensor, gt_depth_map: Tensor, deltas: int,
                reduction: str = "mean") -> Tensor:
  """
  The δ_i error, which is the percentage of relative errors lower than 1.25**i.
  Both inputs should have the same shape, but without any constraint on a specific shape to use.
  Inputs should also *not* contain any NaN value.
  """

  # We initialize the reduction function
  if reduction == "mean":
    reduction_fn = torch.mean
  elif reduction == "sum":
    reduction_fn = torch.sum
  else:
    raise ValueError(f"Unknown reduction '{reduction}'")

  # We compute the maximum relative error for every pixel between pred/gt and gt/pred
  max_rel = torch.maximum(pred_depth_map/gt_depth_map, gt_depth_map/pred_depth_map)

  # We create a tensor of shape (deltas,)
  delta_err = torch.empty(deltas, dtype=torch.float, device=pred_depth_map.device)

  # Then, for each δ, we compute the metric
  for delta in range(deltas):
    delta_err[delta] = reduction_fn((max_rel <= 1.25**(delta+1)).float()).item()

  # And we return it
  return delta_err


def ms_error(pred_depth_map: Tensor, gt_depth_map: Tensor, reduction: str = "mean") -> float:
  """
  The Mean Squared (MS) error, which is the average squared error over the whole depth map.
  Both inputs should have the same shape, but without any constraint on a specific shape to use.
  Inputs should also *not* contain any NaN value.
  """

  # We initialize the reduction function
  if reduction == "mean":
    reduction_fn = torch.mean
  elif reduction == "sum":
    reduction_fn = torch.sum
  else:
    raise ValueError(f"Unknown reduction '{reduction}'")

  # We compute the error on the inputs
  rms_err = reduction_fn((pred_depth_map-gt_depth_map)**2).item()

  # And we return it
  return rms_err


def mslog_error(pred_depth_map: Tensor, gt_depth_map: Tensor, eps: float = 1e-6,
                 reduction: str = "mean") -> float:
  """
  The Mean Squared Log (MSlog) error, which is the average squared error over the whole log depth
  map.
  Both inputs should have the same shape, but without any constraint on a specific shape to use.
  Inputs should also *not* contain any NaN value.
  """

  # We initialize the reduction function
  if reduction == "mean":
    reduction_fn = torch.mean
  elif reduction == "sum":
    reduction_fn = torch.sum
  else:
    raise ValueError(f"Unknown reduction '{reduction}'")

  # We compute the error on the inputs
  rmslog_err = reduction_fn((torch.log(pred_depth_map+eps)-torch.log(gt_depth_map+eps))**2).item()

  # And we return it
  return rmslog_err
