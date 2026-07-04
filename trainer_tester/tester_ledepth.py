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
import json  # New import for JSON output

from fvcore.nn import FlopCountAnalysis
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# Replaced old metrics with new functions
from metrics.metrics_ledepth import abs_rel, rmse_log, si_log, accuracy, mean_absolute_error, textjson

from visualization.visualization import depth_image_to_img, event_volume_to_img, lidar_proj_to_img


class Tester_LEDepth():
    """
    A tester for the  DELTA & LEDepth networks
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
        # We only use the 10m, 20m, and 30m cutoffs for the mean_absolute_error metric
        self.cutoff_dists = (10, 20, 30)

        # We collect the patch size (only for attention-based models)
        # if self.model_name in ("DELTA", "LEDepth", "LEDepth_SSM_UNET"):
        #     self.patch_size = config["patch_size"]
        # else:
        #     self.patch_size = None
        self.patch_size = config["patch_size"]
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
        # New JSON file for final output
        json_file_path = f"out/results/{time_prefix}_results.json"

        # Placeholder for FLOPS calculation
        total_flops = None

        # We initialize the metrics
        # New metrics initialization (bf = before, af = after)
        # We use lists/tensors to accumulate values for 'before' and 'after' depth predictions

        # Before depths (bf) metrics
        total_abs_rel_bf = torch.tensor(0.0, device=self.device)
        total_rmse_log_bf = torch.tensor(0.0, device=self.device)
        total_si_log_bf = torch.tensor(0.0, device=self.device)
        total_acc_delta_bf = torch.zeros(3, dtype=torch.float, device=self.device)
        total_mae_cutoff_bf = torch.zeros(len(self.cutoff_dists), dtype=torch.float, device=self.device)

        # After depths (af) metrics
        total_abs_rel_af = torch.tensor(0.0, device=self.device)
        total_rmse_log_af = torch.tensor(0.0, device=self.device)
        total_si_log_af = torch.tensor(0.0, device=self.device)
        total_acc_delta_af = torch.zeros(3, dtype=torch.float, device=self.device)
        total_mae_cutoff_af = torch.zeros(len(self.cutoff_dists), dtype=torch.float, device=self.device)

        # Counters
        total_samples = 0
        total_runtime = 0.0  # To accumulate inference time

        # For each sequence extracted from the dataset...
        for seq_idx, sequence in enumerate(tqdm(self.test_dataloader, "Testing")):
            # We compute some infos about the length of the sequence
            total_items = len(sequence)
            total_samples += total_items  # Accumulate total samples processed

            # We initialize the memories of the network
            central_mem = None
            prop_mem = None
            ledepth_states = None
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

                nan_mask = torch.isnan(bf_depths)
                bf_depths[nan_mask] = 1
                valid_mask = bf_depths != 1

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
                        flop = FlopCountAnalysis(self.model, (lidar_proj, event_volume, ledepth_states, crop_positions))
                    elif self.model_name == "LEDepth_SSM_UNET":
                        flop = FlopCountAnalysis(self.model, (lidar_proj, event_volume))
                    else:
                        flop = FlopCountAnalysis(self.model, (lidar_proj, event_volume))
                        # raise NotImplementedError(f"Model {self.model_name} is not implemented")

                    total_flops = flop.total()
                    tqdm.write(f"Total FLOPS: {total_flops}")

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
                    pred_depths, student_depth, ledepth_states = self.model(
                        lidar_input=lidar_proj,
                        event_input=event_volume,
                        prev_states=ledepth_states,
                        crop_positions=crop_positions
                    )
                elif self.model_name == "LEDepth_SSM_UNET":
                    pred_depths = self.model(lidar_proj,event_volume)
                else:
                    pred_depths = self.model(lidar_proj, event_volume)
                    # raise NotImplementedError(f"Model {self.model_name} is not implemented")
                pred_depths = pred_depths.float()
                pred_depths[~valid_mask] = 1
                # If inference time is measured, we compute it for this prediction and add it to total runtime
                # Note: a call torch.cuda.synchronize() is mandatory to ensure that the cuda computation is
                # not done asynchronously
                if self.measure_comp_complexity:
                    torch.cuda.synchronize()
                    t_elapsed = time() - t_ref
                    total_runtime += t_elapsed

                # We correct the prediction, to force it to be in the [0, 1] range
                pred_depths[pred_depths < 0.0] = 0.0
                pred_depths[pred_depths > 1.0] = 1.0

                # We remove any padding before using the data further on
                unpadded_pred_depths = pred_depths[:, :, min_y:max_y, min_x:max_x]
                if bf_depths_available:
                    unpadded_bf_depths = bf_depths[:, :, min_y:max_y, min_x:max_x]
                if af_depths_available:
                    unpadded_af_depths = af_depths[:, :, min_y:max_y, min_x:max_x]

                # ... (Image saving code remains the same) ...

                # If required, we save images of the input and output data
                if self.save_viz:
                    # We begin by computing the current index
                    idx = seq_idx * total_items + item_idx

                    # We create the "images" folder if necessary
                    if not os.path.isdir("out/images"):
                        os.mkdir("out/images")

                    # For the LiDAR projection
                    # if lidar_proj_available:
                    #     lidar_img = lidar_proj_to_img(lidar_proj[:, :, min_y:max_y, min_x:max_x])
                    #     save_image(lidar_img, f"out/images/lidar/lidar_{idx:06d}.png")
                    #
                    # # For the RGB image
                    # if rgb_image_available:
                    #     save_image(rgb_image, f"out/images/rgb/rgb_{idx:06d}.png")
                    #
                    # # For the event volume
                    # if event_volume_available:
                    #     events_img = event_volume_to_img(event_volume[:, :, min_y:max_y, min_x:max_x])
                    #     save_image(events_img, f"out/images/event/evts_{idx:06d}.png")
                    #
                    # # For the D_bf depth image
                    # if bf_depths_available:
                    #     bf_depth_image_img = depth_image_to_img(unpadded_bf_depths)
                    #     save_image(bf_depth_image_img, f"out/images/gt/gt_{idx:06d}.png")

                    # For the D_af depth image
                    # if af_depths_available:
                    #     af_depth_image_img = depth_image_to_img(unpadded_af_depths)
                    #     save_image(af_depth_image_img, f"out/images/gtaf/gtaf_{idx:06d}.png")

                    # For the estimated D_bf depths
                    pred_bf_img = depth_image_to_img(unpadded_pred_depths[:, [0], :, :])
                    save_image(pred_bf_img, f"out/images/pred/pred_{idx:06d}.png")

                    # For the estimated D_af depths
                    if self.out_channels == 2:
                        pred_af_img = depth_image_to_img(unpadded_pred_depths[:, [1], :, :])
                        save_image(pred_af_img, f"out/images/preaf/predaf_{idx:06d}.png")

                    # For the error on D_bf
                    # if bf_depths_available:
                    #     error_bf = torch.abs(unpadded_bf_depths - unpadded_pred_depths[:, [0], :, :])
                    #     error_bf[error_bf < 0.5 / self.lidar_max_range] = torch.nan
                    #     error_bf_img = depth_image_to_img(error_bf)
                    #     save_image(error_bf_img, f"out/images/erro/erro_{idx:06d}.png")

                    # # For the error on D_af
                    # if af_depths_available and self.out_channels == 2:
                    #     error_af = torch.abs(unpadded_af_depths - unpadded_pred_depths[:, [1], :, :])
                    #     error_af[error_af < 0.5 / self.lidar_max_range] = torch.nan
                    #     error_af_img = depth_image_to_img(error_af)  # Fixed typo, error_af used here
                    #     save_image(error_af_img, f"out/images/erroaf/erroraf_{idx:06d}.png")

                # --- Metric Calculation (new logic) ---

                # Common preparation: Remove NaNs and convert back to metric values
                if bf_depths_available:
                    not_nan_mask_bf = ~torch.isnan(unpadded_bf_depths)
                    pred_bf = unpadded_pred_depths[:, [0], :, :][not_nan_mask_bf]
                    gt_bf = unpadded_bf_depths[not_nan_mask_bf]
                    metric_pred_bf = pred_bf * self.lidar_max_range
                    metric_gt_bf = gt_bf * self.lidar_max_range

                if self.out_channels == 2 and af_depths_available:
                    not_nan_mask_af = ~torch.isnan(unpadded_af_depths)
                    pred_af = unpadded_pred_depths[:, [1], :, :][not_nan_mask_af]
                    gt_af = unpadded_af_depths[not_nan_mask_af]
                    metric_pred_af = pred_af * self.lidar_max_range
                    metric_gt_af = gt_af * self.lidar_max_range

                # Compute and accumulate metrics for D_bf (before depth)
                if bf_depths_available:
                    total_abs_rel_bf += abs_rel(metric_pred_bf, metric_gt_bf)
                    total_rmse_log_bf += rmse_log(metric_pred_bf, metric_gt_bf)
                    total_si_log_bf += si_log(metric_pred_bf, metric_gt_bf)

                    # Delta accuracies
                    delta_thresholds = [1.25, 1.25 ** 2, 1.25 ** 3]
                    for i, threshold in enumerate(delta_thresholds):
                        total_acc_delta_bf[i] += accuracy(metric_pred_bf, metric_gt_bf, threshold)

                    # Mean Absolute Error at cutoffs
                    mae_errors = mean_absolute_error(metric_pred_bf, metric_gt_bf, self.cutoff_dists)
                    for i in range(len(self.cutoff_dists)):
                        # mean_absolute_error returns a list of mean errors for each cutoff
                        total_mae_cutoff_bf[i] += mae_errors[i]

                # Compute and accumulate metrics for D_af (after depth)
                if self.out_channels == 2 and af_depths_available:
                    total_abs_rel_af += abs_rel(metric_pred_af, metric_gt_af)
                    total_rmse_log_af += rmse_log(metric_pred_af, metric_gt_af)
                    total_si_log_af += si_log(metric_pred_af, metric_gt_af)

                    # Delta accuracies
                    delta_thresholds = [1.25, 1.25 ** 2, 1.25 ** 3]
                    for i, threshold in enumerate(delta_thresholds):
                        total_acc_delta_af[i] += accuracy(metric_pred_af, metric_gt_af, threshold)

                    # Mean Absolute Error at cutoffs
                    mae_errors = mean_absolute_error(metric_pred_af, metric_gt_af, self.cutoff_dists)
                    for i in range(len(self.cutoff_dists)):
                        total_mae_cutoff_af[i] += mae_errors[i]

        # --- Final Output Generation ---

        # Calculate averages
        avg_abs_rel_bf = total_abs_rel_bf / total_samples
        avg_rmse_log_bf = total_rmse_log_bf / total_samples
        avg_si_log_bf = total_si_log_bf / total_samples
        avg_acc_delta_bf = total_acc_delta_bf / total_samples
        avg_mae_cutoff_bf = total_mae_cutoff_bf / total_samples

        # If predicting 'after' depths, calculate their averages too
        if self.out_channels == 2:
            avg_abs_rel_af = total_abs_rel_af / total_samples
            avg_rmse_log_af = total_rmse_log_af / total_samples
            avg_si_log_af = total_si_log_af / total_samples
            avg_acc_delta_af = total_acc_delta_af / total_samples
            avg_mae_cutoff_af = total_mae_cutoff_af / total_samples

        # Total runtime in milliseconds
        avg_runtime_ms = (total_runtime / total_samples) * 1000

        # Build the final JSON object (using bf metrics for the main output)
        final_results = {
            "Abs.Rel": avg_abs_rel_bf,
            "RMSELog": avg_rmse_log_bf,
            "SILog": avg_si_log_bf,
            "delta<1.25": avg_acc_delta_bf[0],
            "delta<1.25^2": avg_acc_delta_bf[1],
            "delta<1.25^3": avg_acc_delta_bf[2],
            "Cutoff(10,20,30)": avg_mae_cutoff_bf,
            "Runtime(ms)": avg_runtime_ms,
            "seed": 123,  # Placeholder as requested
            "flops": total_flops if self.measure_comp_complexity else "Not measured"
        }

        # Add 'after' depth metrics to the JSON if they were calculated
        if self.out_channels == 2:
            final_results["Abs.Rel_af"] = avg_abs_rel_af
            final_results["RMSELog_af"] = avg_rmse_log_af
            final_results["SILog_af"] = avg_si_log_af
            final_results["delta<1.25_af"] = avg_acc_delta_af[0]
            final_results["delta<1.25^2_af"] = avg_acc_delta_af[1]
            final_results["delta<1.25^3_af"] = avg_acc_delta_af[2]
            final_results["Cutoff(10,20,30)_af"] = avg_mae_cutoff_af

        # Write the results to the JSON file
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4, cls=textjson)

        tqdm.write("FINAL ERROR (written to JSON):")
        tqdm.write(json.dumps(final_results, indent=4, cls=textjson))

    # load_model_checkpoint method remains unchanged
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


# python test_ledepth.py configs/LEDepth/test_sled.json /root/autodl-tmp/project/out/saves/20260111_142135/20260111_142135_100.pth