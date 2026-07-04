#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains a tester class, which can be used to test the ALED & DELTA networks, on either
the SLED, the MVSEC, or the M3ED datasets, as described in our "DELTA: Dense Depth from Events and
LiDAR using Transformer's Attention" article (CVPRW 2025).
"""

from datetime import datetime
from math import sqrt
import os
from time import time

from fvcore.nn import FlopCountAnalysis
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from metrics.metrics import absrel_error, delta_error, l1_error, ms_error, mslog_error
from visualization.visualization import depth_image_to_img, event_volume_to_img, lidar_proj_to_img


class Tester():
  """
  A tester for the ALED & DELTA & LEDepth networks
  """

  def __init__(self, model: nn.Module, test_dataloader: DataLoader, device: str, config: dict):
    # We set the model, the dataloader, and the device
    self.model = model
    self.test_dataloader = test_dataloader
    self.device = device

    # We save the name of the model
    self.model_name = config["model"]

    # We collect the lidar_max_range parameter from the config, and compute the cutoff values based
    # on it
    self.lidar_max_range = config["lidar_max_range"]
    self.cutoff_dists = (10, 20, 30, self.lidar_max_range//2, self.lidar_max_range)

    # We collect the patch size (only for attention-based models)
    if self.model_name in ("DELTA", "LEDepth"):
      self.patch_size = config["patch_size"]
    else:
      self.patch_size = None

    # We determine the number of output channels of the model
    self.out_channels = 2 if config["predict_af_depths"] else 1

    # We save whether we should save images of the results or not, and whether we should evaluate
    # the inference time
    self.save_viz = config["save_visualization"]
    self.measure_comp_complexity = config["measure_computational_complexity"]


  @torch.inference_mode()
  def test(self) -> None:
    """
    Run testing on the whole dataset
    """

    # To start, we must not forget to set the model to evaluation mode
    self.model.eval()

    # We create/open our txt files in which we will store the results
    if not os.path.isdir("out/results"):
      os.mkdir("out/results")
    time_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_file_seq = open(f"out/results/{time_prefix}_per_seq.txt", "w", encoding="utf-8")
    txt_file_global = open(f"out/results/{time_prefix}_global.txt", "w", encoding="utf-8")
    if self.measure_comp_complexity:
      txt_file_inf_time = open(f"out/results/{time_prefix}_inf_time.txt", "w", encoding="utf-8")

    # We initialize the metrics
    # We use an array of 5 elements, as we use 5 cutoff distances: 10m, 20m, 30m, half, no cutoff
    total_mean_err_bf = torch.zeros(5, dtype=torch.float, device=self.device)
    total_absrel_err_bf = torch.zeros(5, dtype=torch.float, device=self.device)
    total_delta_err_bf = torch.zeros((5, 3), dtype=torch.float, device=self.device)
    total_ms_err_bf = torch.zeros(5, dtype=torch.float, device=self.device)
    total_mslog_err_bf = torch.zeros(5, dtype=torch.float, device=self.device)
    total_mean_err_af = torch.zeros(5, dtype=torch.float, device=self.device)
    total_absrel_err_af = torch.zeros(5, dtype=torch.float, device=self.device)
    total_delta_err_af = torch.zeros((5, 3), dtype=torch.float, device=self.device)
    total_ms_err_af = torch.zeros(5, dtype=torch.float, device=self.device)
    total_mslog_err_af = torch.zeros(5, dtype=torch.float, device=self.device)
    total_pixels_bf = torch.zeros(5, dtype=torch.int64, device=self.device)
    total_pixels_af = torch.zeros(5, dtype=torch.int64, device=self.device)

    # For each sequence extracted from the dataset...
    for seq_idx, sequence in enumerate(tqdm(self.test_dataloader, "Testing")):
      # We compute some infos about the length of the sequence
      total_items = len(sequence)

      # We initialize the metrics on this sequence
      # As before, we use an array of 5 elements as we use 5 cutoff distances
      seq_mean_err_bf = torch.zeros(5, dtype=torch.float, device=self.device)
      seq_absrel_err_bf = torch.zeros(5, dtype=torch.float, device=self.device)
      seq_delta_err_bf = torch.zeros((5, 3), dtype=torch.float, device=self.device)
      seq_ms_err_bf = torch.zeros(5, dtype=torch.float, device=self.device)
      seq_mslog_err_bf = torch.zeros(5, dtype=torch.float, device=self.device)
      seq_mean_err_af = torch.zeros(5, dtype=torch.float, device=self.device)
      seq_absrel_err_af = torch.zeros(5, dtype=torch.float, device=self.device)
      seq_delta_err_af = torch.zeros((5, 3), dtype=torch.float, device=self.device)
      seq_ms_err_af = torch.zeros(5, dtype=torch.float, device=self.device)
      seq_mslog_err_af = torch.zeros(5, dtype=torch.float, device=self.device)
      seq_pixels_bf = torch.zeros(5, dtype=torch.int64, device=self.device)
      seq_pixels_af = torch.zeros(5, dtype=torch.int64, device=self.device)

      # We initialize the memories of the network
      central_mem = None
      prop_mem = None

      # For each item (1 LiDAR image, 1 RGB image, 1 event volume, 1 "before" depth image,
      # 1 "after" depth image, 1 padding info, 1 cropping info) in the sequence...
      for item_idx, item in enumerate(tqdm(sequence, "Sequence", leave=False)):
        # We extract the data from the sequence, we check if they are available, and we upload
        # them to the device if it is the case
        # We also make sure that the ground truth depths are in the range [0, 1]
        lidar_proj, rgb_image, event_volume, bf_depths, af_depths, pad_positions, crop_positions = item

        lidar_proj_available = not torch.all(torch.isnan(lidar_proj))
        rgb_image_available = not torch.all(torch.isnan(rgb_image))
        event_volume_available = not torch.all(torch.isnan(event_volume))
        bf_depths_available = not torch.all(torch.isnan(bf_depths))
        af_depths_available = not torch.all(torch.isnan(af_depths))

        if lidar_proj_available:
          lidar_proj = lidar_proj.to(self.device, non_blocking=True)
          prop_mem = None
          padded_img_size = lidar_proj.shape[-2:]
        else:
          lidar_proj = None

        if rgb_image_available:
          rgb_image = rgb_image.to(self.device, non_blocking=True)
          padded_img_size = rgb_image.shape[-2:]
        else:
          rgb_image = None

        if event_volume_available:
          event_volume = event_volume.to(self.device, non_blocking=True)
          padded_img_size = event_volume.shape[-2:]
        else:
          event_volume = None

        if bf_depths_available:
          bf_depths = bf_depths.to(self.device, non_blocking=True)
          bf_depths[bf_depths > 1.0] = 1.0
        else:
          bf_depths = None

        if af_depths_available:
          af_depths = af_depths.to(self.device, non_blocking=True)
          af_depths[af_depths > 1.0] = 1.0
        else:
          af_depths = None

        crop_positions = crop_positions.to(self.device, non_blocking=True)
        if self.patch_size is not None:
          crop_positions //= self.patch_size

        # We compute the coordinates to use to remove the padding
        pad_top, pad_bottom, pad_left, pad_right = pad_positions[0, :]
        min_x = pad_left
        max_x = padded_img_size[1] - pad_right
        min_y = pad_top
        max_y = padded_img_size[0] - pad_bottom

        # If necessary, we compute the FLOPS and display them
        if self.measure_comp_complexity and seq_idx == 0 and item_idx == 0:
          if self.model_name == "ALED":
            flop = FlopCountAnalysis(self.model, (lidar_proj, event_volume, central_mem))
          elif self.model_name == "DELTA":
            flop = FlopCountAnalysis(self.model, (lidar_proj, event_volume, central_mem, prop_mem,
                                                  crop_positions))
          elif self.model_name == "LEDepth":
            flop = FlopCountAnalysis(self.model, (lidar_proj, event_volume, central_mem, prop_mem,
                                                  crop_positions))
          else:
            raise NotImplementedError(f"Model {self.model_name} is not implemented")

          tqdm.write(f"Total FLOPS: {flop.total()}")

        # If inference time is measured, we set our reference time
        if self.measure_comp_complexity:
          t_ref = time()

        # We run a prediction
        if self.model_name == "ALED":
          pred_depths, central_mem = self.model(lidar_proj, event_volume, central_mem)
        elif self.model_name == "DELTA":
          pred_depths, central_mem, prop_mem = self.model(lidar_proj, event_volume, central_mem,
                                                          prop_mem, crop_positions)
        elif self.model_name == "LEDepth":
          pred_depths, central_mem, prop_mem = self.model(lidar_proj, event_volume, central_mem,
                                                          prop_mem, crop_positions)
        else:
          raise NotImplementedError(f"Model {self.model_name} is not implemented")

        # If inference time is measured, we compute it for this prediction and write it in the
        # output file
        # Note: a call torch.cuda.synchronize() is mandatory to ensure that the cuda computation is
        # not done asynchronously
        if self.measure_comp_complexity:
          torch.cuda.synchronize()
          t_elapsed = time() - t_ref
          txt_file_inf_time.write(f"{t_elapsed}\n")

        # We correct the prediction, to force it to be in the [0, 1] range
        pred_depths[pred_depths < 0.0] = 0.0
        pred_depths[pred_depths > 1.0] = 1.0

        # We remove any padding before using the data further on
        if bf_depths_available:
          unpadded_bf_depths = bf_depths[:, :, min_y:max_y, min_x:max_x]
        if af_depths_available:
          unpadded_af_depths = af_depths[:, :, min_y:max_y, min_x:max_x]
        unpadded_pred_depths = pred_depths[:, :, min_y:max_y, min_x:max_x]

        # If required, we save images of the input and output data
        if self.save_viz:
          # We begin by computing the current index
          idx = seq_idx*total_items + item_idx

          # We create the "images" folder if necessary
          if not os.path.isdir("out/images"):
            os.mkdir("out/images")

          # For the LiDAR projection
          if lidar_proj_available:
            lidar_img = lidar_proj_to_img(lidar_proj[:, :, min_y:max_y, min_x:max_x])
            save_image(lidar_img, f"out/images/lidar_{idx:06d}.png")

          # For the RGB image
          if rgb_image_available:
            save_image(rgb_image, f"out/images/rgb_{idx:06d}.png")

          # For the event volume
          if event_volume_available:
            events_img = event_volume_to_img(event_volume[:, :, min_y:max_y, min_x:max_x])
            save_image(events_img, f"out/images/evts_{idx:06d}.png")

          # For the D_bf depth image
          if bf_depths_available:
            bf_depth_image_img = depth_image_to_img(unpadded_bf_depths)
            save_image(bf_depth_image_img, f"out/images/gtbf_{idx:06d}.png")

          # For the D_af depth image
          if af_depths_available:
            af_depth_image_img = depth_image_to_img(unpadded_af_depths)
            save_image(af_depth_image_img, f"out/images/gtaf_{idx:06d}.png")

          # For the estimated D_bf depths
          pred_bf_img = depth_image_to_img(unpadded_pred_depths[:, [0], :, :])
          save_image(pred_bf_img, f"out/images/predbf_{idx:06d}.png")

          # For the estimated D_af depths
          if self.out_channels == 2:
            pred_af_img = depth_image_to_img(unpadded_pred_depths[:, [1], :, :])
            save_image(pred_af_img, f"out/images/predaf_{idx:06d}.png")

          # For the error on D_bf
          if bf_depths_available:
            error_bf = torch.abs(unpadded_bf_depths - unpadded_pred_depths[:, [0], :, :])
            error_bf[error_bf < 0.5/self.lidar_max_range] = torch.nan
            error_bf_img = depth_image_to_img(error_bf)
            save_image(error_bf_img, f"out/images/errorbf_{idx:06d}.png")

          # For the error on D_af
          if af_depths_available and self.out_channels == 2:
            error_af = torch.abs(unpadded_af_depths - unpadded_pred_depths[:, [1], :, :])
            error_af[error_af < 0.5/self.lidar_max_range] = torch.nan
            error_af_img = depth_image_to_img(error_bf)
            save_image(error_af_img, f"out/images/erroraf_{idx:06d}.png")

        # Removing NaN values for D_bf, and converting them back to metric values
        if bf_depths_available:
          not_nan_mask_bf = ~torch.isnan(unpadded_bf_depths)
          masked_unpadded_pred_bf = unpadded_pred_depths[:, [0], :, :][not_nan_mask_bf]
          masked_unpadded_bf_depths = unpadded_bf_depths[not_nan_mask_bf]
          metric_masked_unpadded_pred_bf = masked_unpadded_pred_bf * self.lidar_max_range
          metric_masked_unpadded_bf_depths = masked_unpadded_bf_depths * self.lidar_max_range

        # Removing NaN values for D_af, and converting them back to metric values
        if self.out_channels == 2 and af_depths_available:
          not_nan_mask_af = ~torch.isnan(unpadded_af_depths)
          masked_unpadded_pred_af = unpadded_pred_depths[:, [1], :, :][not_nan_mask_af]
          masked_unpadded_af_depths = unpadded_af_depths[not_nan_mask_af]
          metric_masked_unpadded_pred_af = masked_unpadded_pred_af * self.lidar_max_range
          metric_masked_unpadded_af_depths = masked_unpadded_af_depths * self.lidar_max_range

        # Then, for each cutoff distance...
        for c, cutoff in enumerate(self.cutoff_dists):
          # We compute the errors for D_bf
          if bf_depths_available:
            # We apply the cutoff as a mask on the ground truth and the prediction
            cutoff_mask_bf = metric_masked_unpadded_bf_depths <= cutoff
            cutoff_pred_bf = metric_masked_unpadded_pred_bf[cutoff_mask_bf]
            cutoff_bf_depths = metric_masked_unpadded_bf_depths[cutoff_mask_bf]

            # Mean error on D_bf
            seq_mean_err_bf[c] += l1_error(cutoff_pred_bf, cutoff_bf_depths, reduction="sum")

            # Absolute relative (AbsRel) error on D_bf
            seq_absrel_err_bf[c] += absrel_error(cutoff_pred_bf, cutoff_bf_depths, reduction="sum")

            # δ1, δ2, and δ3 errors on D_bf
            seq_delta_err_bf[c] += delta_error(cutoff_pred_bf, cutoff_bf_depths, 3, reduction="sum")

            # Mean squared (MS) error on D_bf
            seq_ms_err_bf[c] += ms_error(cutoff_pred_bf, cutoff_bf_depths, reduction="sum")

            # Mean squared log (MSlog) error on D_bf
            seq_mslog_err_bf[c] += mslog_error(cutoff_pred_bf, cutoff_bf_depths, reduction="sum")

            # We count the number of valid pixels that were taken into account
            seq_pixels_bf[c] += torch.sum(cutoff_mask_bf)

          # We compute the errors for D_af
          if self.out_channels == 2 and af_depths_available:
            # We apply the cutoff as a mask on the ground truth and the prediction
            cutoff_mask_af = metric_masked_unpadded_af_depths <= cutoff
            cutoff_pred_af = metric_masked_unpadded_pred_af[cutoff_mask_af]
            cutoff_af_depths = metric_masked_unpadded_af_depths[cutoff_mask_af]

            # Mean error on D_af
            seq_mean_err_af[c] += l1_error(cutoff_pred_af, cutoff_af_depths, reduction="sum")

            # Absolute relative (AbsRel) error on D_af
            seq_absrel_err_af[c] += absrel_error(cutoff_pred_af, cutoff_af_depths, reduction="sum")

            # δ1, δ2, and δ3 errors on D_af
            seq_delta_err_af[c] += delta_error(cutoff_pred_af, cutoff_af_depths, 3, reduction="sum")

            # Mean squared (MS) error on D_af
            seq_ms_err_af[c] += ms_error(cutoff_pred_af, cutoff_af_depths, reduction="sum")

            # Mean squared log (MSlog) error on D_af
            seq_mslog_err_af[c] += mslog_error(cutoff_pred_af, cutoff_af_depths, reduction="sum")

            # We count the number of valid pixels that were taken into account
            seq_pixels_af[c] += torch.sum(cutoff_mask_af)

      # Once the sequence is over, we display the error for each cutoff distance
      seq_name = os.path.split(self.test_dataloader.dataset.sequences_paths[seq_idx])[-1]
      tqdm.write(seq_name)
      for c, cutoff in enumerate(self.cutoff_dists):
        tqdm.write(f"Mean err. bf {cutoff}m: {seq_mean_err_bf[c]/seq_pixels_bf[c]:.3f}; " +
                   f"AbsRel err. bf {cutoff}m: {seq_absrel_err_bf[c]/seq_pixels_bf[c]:.3f}; " +
                   f"δ1 err. bf {cutoff}m: {seq_delta_err_bf[c][0]/seq_pixels_bf[c]:.3f}; " +
                   f"δ2 err. bf {cutoff}m: {seq_delta_err_bf[c][1]/seq_pixels_bf[c]:.3f}; " +
                   f"δ3 err. bf {cutoff}m: {seq_delta_err_bf[c][2]/seq_pixels_bf[c]:.3f}; " +
                   f"RMS err. bf {cutoff}m: {sqrt(seq_ms_err_bf[c]/seq_pixels_bf[c]):.3f}; " +
                   f"RMSlog err. bf {cutoff}m: {sqrt(seq_mslog_err_bf[c]/seq_pixels_bf[c]):.3f}; " +
                   f"Mean err. af {cutoff}m: {seq_mean_err_af[c]/seq_pixels_af[c]:.3f}; " +
                   f"AbsRel err. af {cutoff}m: {seq_absrel_err_af[c]/seq_pixels_af[c]:.3f}; " +
                   f"δ1 err. af {cutoff}m: {seq_delta_err_af[c][0]/seq_pixels_af[c]:.3f}; " +
                   f"δ2 err. af {cutoff}m: {seq_delta_err_af[c][1]/seq_pixels_af[c]:.3f}; " +
                   f"δ3 err. af {cutoff}m: {seq_delta_err_af[c][2]/seq_pixels_af[c]:.3f}; " +
                   f"RMS err. af {cutoff}m: {sqrt(seq_ms_err_af[c]/seq_pixels_af[c]):.3f}; " +
                   f"RMSlog err. af {cutoff}m: {sqrt(seq_mslog_err_af[c]/seq_pixels_af[c]):.3f}")

      # We also write them in the .txt file
      seq_name_latex = seq_name[:-11].replace('_', r'\_')
      txt_file_seq.write(fr"\multirow{{5}}{{*}}{{{seq_name_latex}}} ")
      for c, cutoff in enumerate(self.cutoff_dists):
        txt_file_seq.write(f"& {int(cutoff)}m " +
                           f"& {seq_mean_err_bf[c]/seq_pixels_bf[c]:.3f} " +
                           f"& {seq_absrel_err_bf[c]/seq_pixels_bf[c]:.3f} " +
                           f"& {seq_delta_err_bf[c][0]/seq_pixels_bf[c]:.3f} " +
                           f"& {seq_delta_err_bf[c][1]/seq_pixels_bf[c]:.3f} " +
                           f"& {seq_delta_err_bf[c][2]/seq_pixels_bf[c]:.3f} " +
                           f"& {sqrt(seq_ms_err_bf[c]/seq_pixels_bf[c]):.3f} " +
                           f"& {sqrt(seq_mslog_err_bf[c]/seq_pixels_bf[c]):.3f} " +
                           f"& {seq_mean_err_af[c]/seq_pixels_af[c]:.3f} " +
                           f"& {seq_absrel_err_af[c]/seq_pixels_af[c]:.3f} " +
                           f"& {seq_delta_err_af[c][0]/seq_pixels_af[c]:.3f} " +
                           f"& {seq_delta_err_af[c][1]/seq_pixels_af[c]:.3f} " +
                           f"& {seq_delta_err_af[c][2]/seq_pixels_af[c]:.3f} " +
                           f"& {sqrt(seq_ms_err_af[c]/seq_pixels_af[c]):.3f} " +
                           f"& {sqrt(seq_mslog_err_af[c]/seq_pixels_af[c]):.3f} " +
                           r"\\" + "\n")
      txt_file_seq.write(r"\midrule" + "\n")

      # And we add each error of the sequence to the global error
      for c, cutoff in enumerate(self.cutoff_dists):
        total_mean_err_bf[c] += seq_mean_err_bf[c]
        total_absrel_err_bf[c] += seq_absrel_err_bf[c]
        total_delta_err_bf[c] += seq_delta_err_bf[c]
        total_ms_err_bf[c] += seq_ms_err_bf[c]
        total_mslog_err_bf[c] += seq_mslog_err_bf[c]
        total_mean_err_af[c] += seq_mean_err_af[c]
        total_absrel_err_af[c] += seq_absrel_err_af[c]
        total_delta_err_af[c] += seq_delta_err_af[c]
        total_ms_err_af[c] += seq_ms_err_af[c]
        total_mslog_err_af[c] += seq_mslog_err_af[c]
        total_pixels_bf[c] += seq_pixels_bf[c]
        total_pixels_af[c] += seq_pixels_af[c]

    # Once we have gone over all the sequences, we display the final errors for each cutoff distance
    tqdm.write("FINAL ERROR")
    for c, cutoff in enumerate(self.cutoff_dists):
      tqdm.write(f"Mean err. bf {cutoff}m: {total_mean_err_bf[c]/total_pixels_bf[c]:.2f}; " +
                 f"AbsRel err. bf {cutoff}m: {total_absrel_err_bf[c]/total_pixels_bf[c]:.3f}; " +
                 f"δ1 err. bf {cutoff}m: {total_delta_err_bf[c][0]/total_pixels_bf[c]:.3f}; " +
                 f"δ2 err. bf {cutoff}m: {total_delta_err_bf[c][1]/total_pixels_bf[c]:.3f}; " +
                 f"δ3 err. bf {cutoff}m: {total_delta_err_bf[c][2]/total_pixels_bf[c]:.3f}; " +
                 f"RMS err. bf {cutoff}m: {sqrt(total_ms_err_bf[c]/total_pixels_bf[c]):.3f}; " +
                 f"RMSlog err. bf {cutoff}m: {sqrt(total_mslog_err_bf[c]/total_pixels_bf[c]):.3f}; " +
                 f"Mean err. af {cutoff}m: {total_mean_err_af[c]/total_pixels_af[c]:.2f}; " +
                 f"AbsRel err. af {cutoff}m: {total_absrel_err_af[c]/total_pixels_af[c]:.3f}; " +
                 f"δ1 err. af {cutoff}m: {total_delta_err_af[c][0]/total_pixels_af[c]:.3f}; " +
                 f"δ2 err. af {cutoff}m: {total_delta_err_af[c][1]/total_pixels_af[c]:.3f}; " +
                 f"δ3 err. af {cutoff}m: {total_delta_err_af[c][2]/total_pixels_af[c]:.3f}; " +
                 f"RMS err. af {cutoff}m: {sqrt(total_ms_err_af[c]/total_pixels_af[c]):.3f}; " +
                 f"RMSlog err. af {cutoff}m: {sqrt(total_mslog_err_af[c]/total_pixels_af[c]):.3f}")

    # We also write them in the .txt file
    for c, cutoff in enumerate(self.cutoff_dists):
      txt_file_global.write(f"FINAL ERROR {cutoff}m: " +
                            f"Mean err. bf: {total_mean_err_bf[c]/total_pixels_bf[c]:.2f}; " +
                            f"AbsRel err. bf: {total_absrel_err_bf[c]/total_pixels_bf[c]:.3f}; " +
                            f"δ1 err. bf: {total_delta_err_bf[c][0]/total_pixels_bf[c]:.3f}; " +
                            f"δ2 err. bf: {total_delta_err_bf[c][1]/total_pixels_bf[c]:.3f}; " +
                            f"δ3 err. bf: {total_delta_err_bf[c][2]/total_pixels_bf[c]:.3f}; " +
                            f"RMS err. bf: {sqrt(total_ms_err_bf[c]/total_pixels_bf[c]):.3f}; " +
                            f"RMSlog err. bf: {sqrt(total_mslog_err_bf[c]/total_pixels_bf[c]):.3f}; " +
                            f"Mean err. af: {total_mean_err_af[c]/total_pixels_af[c]:.2f}; " +
                            f"AbsRel err. af: {total_absrel_err_af[c]/total_pixels_af[c]:.3f}; " +
                            f"δ1 err. af: {total_delta_err_af[c][0]/total_pixels_af[c]:.3f}; " +
                            f"δ2 err. af: {total_delta_err_af[c][1]/total_pixels_af[c]:.3f}; " +
                            f"δ3 err. af: {total_delta_err_af[c][2]/total_pixels_af[c]:.3f}; " +
                            f"RMS err. af: {sqrt(total_ms_err_af[c]/total_pixels_af[c]):.3f}; " +
                            f"RMSlog err. af: {sqrt(total_mslog_err_af[c]/total_pixels_af[c]):.3f}\n")

    # And we don't forget to close the .txt files!
    txt_file_seq.close()
    txt_file_global.close()
    if self.measure_comp_complexity:
      txt_file_inf_time.close()


  def load_model_checkpoint(self, path_to_checkpoint: str) -> None:
    """
    Utility function, used for loading the pretrained model from a checkpoint
    """

    # We load the state dict
    state_dict = torch.load(path_to_checkpoint, weights_only=True)

    # We have to remove the "module." prefix if present since the model was probably trained with DP
    # or DDP but is loaded here without any of them
    nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")

    # We finally load the state dict into the model
    self.model.load_state_dict(state_dict)
