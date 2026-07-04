#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains a trainer class, which can be used to train, finetune, and validate the ALED,
DELTA, LEDepth, and LEDEPTH_SSM_UNET networks.
"""

import os

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from metrics.metrics import l1_error
from visualization.visualization import depth_image_to_img, event_volume_to_img, lidar_proj_to_img


class Trainer():
    """
    A trainer for the ALED & DELTA & LEDepth & LEDEPTH_SSM_UNET networks
    """

    def __init__(self, model: DistributedDataParallel, train_dataloader: DataLoader,
                 val_dataloader: DataLoader | None, loss_criterion: nn.Module, optimizer: Optimizer,
                 tensorboard_writer: SummaryWriter | None, config: dict):
        # We set the model, the number of epochs, the dataloaders, the loss criterion, the optimizer,
        # the Tensorboard writer, and the config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = loss_criterion
        self.optimizer = optimizer
        self.writer = tensorboard_writer

        # We also initialize the scaler (due to the use of AMP)
        self.scaler = torch.cuda.amp.GradScaler()

        # We identify the GPU/device in use
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.device = f"cuda:{self.gpu_id}"

        # We save the name of the model
        self.model_name = config["model"]

        # We collect the total number of epochs
        self.num_epochs = config["epochs"]

        # We collect the patch size (only for attention-based models or models needing cropping info)
        if self.model_name == "ALED":
            self.patch_size = None
        else:
            self.patch_size = config["patch_size"]

        # We determine the number of output channels of the model
        self.out_channels = 2 if config["predict_af_depths"] else 1

        # We save the weight for the MSG loss for the first epoch
        self.weight_msg_epoch_0 = config["weight_msg_epoch_0"]

        # [LEDepth] Student Loss Configuration
        # Defaults to False if not provided in config
        self.use_student_loss = config.get("use_student_loss", False)
        # Weight for SSPL loss (can be parameterized in config, default 1.0 or 0.2)
        self.weight_sspl = config.get("weight_sspl", 0.2)

        # We identify the number of accumulation steps to use during the training
        self.accumulation_steps_train = config["accumulation_steps_train"]

        # We get the "how often should we display the losses in the terminal" parameter
        self.losses_display_every_x = config["losses_display_every_x"]

        # We identify the number of training and validation sequences
        self.total_nb_seq_train = len(train_dataloader)
        if val_dataloader is not None:
            self.total_nb_seq_val = len(val_dataloader)
        else:
            self.total_nb_seq_val = 0

    def _detach_state_dict(self, states):
        """
        Helper to detach gradients from the state dictionary (for LEDepth/Mamba)
        """
        if states is None:
            return None
        new_states = {}
        for k, v in states.items():
            # v is a list of tensors or Nones
            if isinstance(v, list):
                new_states[k] = [t.detach() if isinstance(t, torch.Tensor) else t for t in v]
            else:
                new_states[k] = v
        return new_states

    def train(self, epoch: int) -> None:
        """
        Run a single epoch of training
        """

        # We set the model to training mode
        self.model.train()

        # We set to zero the running losses (used for display in the terminal)
        running_bf_loss_l1 = 0.0
        running_bf_loss_ms = 0.0
        running_af_loss_l1 = 0.0
        running_af_loss_ms = 0.0
        running_loss_student = 0.0  # [LEDepth] Track Student loss
        running_loss = 0.0
        running_len = 0

        # We set the weights for the loss
        weight_l1 = 1.0
        if epoch == 0:
            weight_ms = self.weight_msg_epoch_0
        else:
            weight_ms = 1.0

        # For each sequence extracted from the dataset...
        for seq_idx, sequence in enumerate(tqdm(self.train_dataloader, "Training", leave=False,
                                                disable=self.gpu_id != 0)):
            # We set to zero the losses for the sequence (used for display in Tensorboard)
            seq_bf_loss_l1 = 0.0
            seq_bf_loss_ms = 0.0
            seq_af_loss_l1 = 0.0
            seq_af_loss_ms = 0.0
            seq_loss_student = 0.0  # [LEDepth]
            seq_loss = 0.0
            seq_len = 0

            # We initialize the memories based on model type
            central_mem = None  # For ALED/DELTA
            prop_mem = None  # For DELTA
            ledepth_states = None  # For LEDepth (Dict)

            # We initialize the loss for the sequence
            loss = 0

            # For each item in the sequence...
            for item_idx, item in enumerate(sequence):
                lidar_proj, rgb_image, event_volume, bf_depths, af_depths, _, crop_positions = item
                lidar_proj_available = not torch.all(torch.isnan(lidar_proj))
                rgb_image_available = not torch.all(torch.isnan(rgb_image))
                event_volume_available = not torch.all(torch.isnan(event_volume))
                bf_depths_available = not torch.all(torch.isnan(bf_depths))
                af_depths_available = not torch.all(torch.isnan(af_depths))

                if lidar_proj_available:
                    lidar_proj = lidar_proj.to(self.device, non_blocking=True)
                    # Reset memory when new LiDAR comes (Key mechanic for DELTA/ALED)
                    prop_mem = None
                else:
                    lidar_proj = None

                if rgb_image_available:
                    rgb_image = rgb_image.to(self.device, non_blocking=True)
                else:
                    rgb_image = None

                if event_volume_available:
                    event_volume = event_volume.to(self.device, non_blocking=True)
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

                # We enter the mixed precision mode
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    student_depth = None  # Initialize

                    if self.model_name == "ALED":
                        pred_depths, central_mem = self.model(lidar_proj, event_volume, central_mem)

                    elif self.model_name == "DELTA":
                        pred_depths, central_mem, prop_mem = self.model(lidar_proj, event_volume, central_mem,
                                                                        prop_mem, crop_positions)

                    elif self.model_name == "LEDepth":
                        # LEDepth returns: final_depth, student_depth, new_states
                        pred_depths, student_depth, ledepth_states = self.model(
                            lidar_proj, event_volume, ledepth_states, crop_positions
                        )

                    # elif self.model_name == "LEDEPTH_SSM_UNET":
                    #     pred_depths = self.model(lidar_proj, event_volume)

                    else:
                        pred_depths = self.model(lidar_proj, event_volume)
                        # raise NotImplementedError(f"Model {self.model_name} is not implemented")

                    # --- Main Losses ---
                    if bf_depths_available:
                        # pred_depths shape: (B, C, H, W). We take channel 0.
                        bf_loss_l1, bf_loss_ms = self.criterion(pred_depths[:, [0], :, :], bf_depths)
                    else:
                        bf_loss_l1, bf_loss_ms = torch.tensor(0.0, device=self.device), torch.tensor(0.0,
                                                                                                     device=self.device)

                    if self.out_channels == 2 and af_depths_available:
                        # If predicting After-Frame depth (channel 1)
                        af_loss_l1, af_loss_ms = self.criterion(pred_depths[:, [1], :, :], af_depths)
                    else:
                        af_loss_l1, af_loss_ms = torch.tensor(0.0, device=self.device), torch.tensor(0.0,
                                                                                                     device=self.device)

                    # --- [LEDepth] Student Loss (Optional) ---
                    loss_student = torch.tensor(0.0, device=self.device)
                    # We calculate this loss if we have a student prediction AND valid sparse LiDAR to supervise it
                    if self.model_name == "LEDepth" and student_depth is not None and lidar_proj is not None:
                        # Teacher: Sparse LiDAR (lidar_proj)
                        # Student: Event-only prediction (student_depth)
                        valid_mask = (lidar_proj > 0)
                        # Calculate loss only on valid pixels
                        if valid_mask.sum() > 0:
                            # Standard L1 loss on sparse points
                            loss_student_val = torch.abs(student_depth[valid_mask] - lidar_proj[valid_mask]).mean()

                            # Only add to gradients if configured
                            if self.use_student_loss:
                                loss_student = loss_student_val * self.weight_sspl
                            else:
                                # Just for logging
                                loss_student = loss_student_val.detach()

                    # Complete loss
                    loss_ = weight_l1 * (bf_loss_l1 + af_loss_l1) + \
                            weight_ms * (bf_loss_ms + af_loss_ms)

                    # Add student loss if enabled (already weighted above)
                    if self.use_student_loss:
                        loss_ += loss_student

                    loss += loss_

                    # Logging
                    if self.gpu_id == 0:
                        seq_bf_loss_l1 += bf_loss_l1.item()
                        seq_bf_loss_ms += bf_loss_ms.item()
                        seq_af_loss_l1 += af_loss_l1.item()
                        seq_af_loss_ms += af_loss_ms.item()
                        seq_loss_student += loss_student.item()
                        seq_loss += loss_.item()
                        seq_len += 1

                        running_bf_loss_l1 += bf_loss_l1.item()
                        running_bf_loss_ms += bf_loss_ms.item()
                        running_af_loss_l1 += af_loss_l1.item()
                        running_af_loss_ms += af_loss_ms.item()
                        running_loss_student += loss_student.item()
                        running_loss += loss_.item()
                        running_len += 1

                    # Gradient step
                    if (item_idx + 1) % self.accumulation_steps_train == 0:
                        with torch.autocast("cuda", enabled=False):
                            # We apply the backwards pass
                            self.optimizer.zero_grad()
                            self.scaler.scale(loss).backward()

                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                            self.scaler.step(self.optimizer)
                            self.scaler.update()

                        # Detach memories/states
                        if isinstance(central_mem, list):
                            central_mem = [mem.detach() for mem in central_mem]
                        elif central_mem is not None:
                            central_mem = central_mem.detach()

                        # [LEDepth] Detach state dict
                        if ledepth_states is not None:
                            ledepth_states = self._detach_state_dict(ledepth_states)

                        loss = 0

            # Tensorboard Logging (End of Sequence)
            if self.gpu_id == 0:
                curr_idx = epoch * self.total_nb_seq_train + seq_idx
                self.writer.add_scalar("Loss/BF_L1", seq_bf_loss_l1 / seq_len, curr_idx)
                self.writer.add_scalar("Loss/BF_MS", seq_bf_loss_ms / seq_len, curr_idx)
                self.writer.add_scalar("Loss/AF_L1", seq_af_loss_l1 / seq_len, curr_idx)
                self.writer.add_scalar("Loss/AF_MS", seq_af_loss_ms / seq_len, curr_idx)
                self.writer.add_scalar("Loss/Student", seq_loss_student / seq_len, curr_idx)
                self.writer.add_scalar("Loss/Total", seq_loss / seq_len, curr_idx)

                # Terminal Display
                if (seq_idx + 1) % self.losses_display_every_x == 0:
                    tqdm.write(f"Epoch {epoch + 1}/{self.num_epochs}, "
                               f"Seq {seq_idx + 1}/{self.total_nb_seq_train}, "
                               f"L1: {running_bf_loss_l1 / running_len:.4f}, "
                               f"MS: {running_bf_loss_ms / running_len:.4f}, "
                               f"Student: {running_loss_student / running_len:.4f}, "
                               f"Total: {running_loss / running_len:.4f}")
                    # Reset running counters
                    running_bf_loss_l1 = 0.0
                    running_bf_loss_ms = 0.0
                    running_af_loss_l1 = 0.0
                    running_af_loss_ms = 0.0
                    running_loss_student = 0.0
                    running_loss = 0.0
                    running_len = 0

    @torch.inference_mode()
    def val(self, epoch: int) -> None:
        """
        Run validation of the model.
        """
        self.model.eval()
        running_val_bf_error = 0.0
        running_val_af_error = 0.0
        nb_bf_errors = 0
        nb_af_errors = 0

        for seq_idx, sequence in enumerate(tqdm(self.val_dataloader, "Validation", leave=False)):
            total_items = len(sequence)

            # Init memories
            central_mem = None
            prop_mem = None
            ledepth_states = None

            for item_idx, item in enumerate(tqdm(sequence, "Sequence", leave=False)):
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

                # Padding info
                pad_top, pad_bottom, pad_left, pad_right = pad_positions[0, :]
                min_x = pad_left
                max_x = padded_img_size[1] - pad_right
                min_y = pad_top
                max_y = padded_img_size[0] - pad_bottom

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    student_depth = None

                    if self.model_name == "ALED":
                        pred_depths, central_mem = self.model.module(lidar_proj, event_volume, central_mem)

                    elif self.model_name == "DELTA":
                        # 用于ledepeth.py
                        pred_depths, central_mem, prop_mem = self.model.module(lidar_proj, event_volume,
                                                                               central_mem, prop_mem,
                                                                               crop_positions)

                    elif self.model_name == "LEDepth":
                        pred_depths, student_depth, ledepth_states = self.model.module(
                            lidar_proj, event_volume, ledepth_states, crop_positions
                        )

                    # elif self.model_name == "LEDEPTH_SSM_UNET":
                    #     pred_depths = self.model.module(lidar_proj, event_volume)

                    else:
                        pred_depths = self.model.module(lidar_proj, event_volume)

                pred_depths = pred_depths.float()
                if student_depth is not None:
                    student_depth = student_depth.float()
                # Clamp Predictions
                pred_depths = torch.clamp(pred_depths, 0.0, 1.0)

                # Unpadding
                unpadded_pred_depths = pred_depths[:, :, min_y:max_y, min_x:max_x]
                if bf_depths_available:
                    unpadded_bf_depths = bf_depths[:, :, min_y:max_y, min_x:max_x]
                if af_depths_available:
                    unpadded_af_depths = af_depths[:, :, min_y:max_y, min_x:max_x]

                # [LEDepth] Process Student Depth for Visualization
                unpadded_student_depth = None
                if student_depth is not None:
                    student_depth = torch.clamp(student_depth, 0.0, 1.0)
                    unpadded_student_depth = student_depth[:, :, min_y:max_y, min_x:max_x]

                # --- Visualization ---
                idx = epoch * self.total_nb_seq_val * total_items + seq_idx * total_items + item_idx

                if lidar_proj_available:
                    lidar_img = lidar_proj_to_img(lidar_proj[:, :, min_y:max_y, min_x:max_x])
                    self.writer.add_images("LiDAR", lidar_img, idx)

                if rgb_image_available:
                    self.writer.add_images("RGB", rgb_image, idx)

                if event_volume_available:
                    events_img = event_volume_to_img(event_volume[:, :, min_y:max_y, min_x:max_x])
                    self.writer.add_images("Events", events_img, idx)

                if bf_depths_available:
                    bf_depth_image_img = depth_image_to_img(unpadded_bf_depths)
                    self.writer.add_images("GT/BF", bf_depth_image_img, idx)

                # Final Prediction
                pred_bf_img = depth_image_to_img(unpadded_pred_depths[:, [0], :, :])
                self.writer.add_images("Pred/BF", pred_bf_img, idx)

                # [LEDepth] Student Visualization
                if unpadded_student_depth is not None:
                    student_img = depth_image_to_img(unpadded_student_depth)
                    self.writer.add_images("Pred/Student", student_img, idx)

                # --- Metrics ---
                if bf_depths_available:
                    not_nan_mask_bf = ~torch.isnan(unpadded_bf_depths)
                    masked_pred = unpadded_pred_depths[:, [0], :, :][not_nan_mask_bf]
                    masked_gt = unpadded_bf_depths[not_nan_mask_bf]
                    running_val_bf_error += l1_error(masked_pred, masked_gt)
                    nb_bf_errors += 1

                if self.out_channels == 2 and af_depths_available:
                    not_nan_mask_af = ~torch.isnan(unpadded_af_depths)
                    masked_pred = unpadded_pred_depths[:, [1], :, :][not_nan_mask_af]
                    masked_gt = unpadded_af_depths[not_nan_mask_af]
                    running_val_af_error += l1_error(masked_pred, masked_gt)
                    nb_af_errors += 1

        # Summary Metrics
        val_bf_error = running_val_bf_error / max(nb_bf_errors, 1)
        val_af_error = running_val_af_error / max(nb_af_errors, 1)

        self.writer.add_scalar("Val/Error_BF", val_bf_error, epoch)
        self.writer.add_scalar("Val/Error_AF", val_af_error, epoch)
        self.writer.add_scalar("Val/Total", (val_bf_error + val_af_error) / 2, epoch)

    def load_model_checkpoint(self, path_to_checkpoint: str) -> None:
        """
        Utility function, used for loading the pretrained model from a checkpoint
        """

        # As we use DDP, we must remap storage from GPU 0 to the local GPU
        map_location = {"cuda:0": f"cuda:{self.gpu_id}"}

        # We load the state dict using the remapping
        state_dict = torch.load(path_to_checkpoint, map_location=map_location, weights_only=True)

        # We check that the checkpoint was trained with DDP, if not we add the "module." prefix
        state_dict_keys = list(state_dict.keys())
        for key in state_dict_keys:
            if not key.startswith("module."):
                state_dict[f"module.{key}"] = state_dict.pop(key)

        # We finally load the state dict into the model
        self.model.load_state_dict(state_dict)

    def save_model_checkpoint(self, time_prefix: str, epoch: int) -> None:
        """
        Utility function, used for saving the pretrained model to a checkpoint
        """
        save_interval = 20
        if (epoch+1) % save_interval != 0:
            return
        # We create the "saves" folder if necessary
        if not os.path.isdir("out/saves"):
            os.mkdir("out/saves")

        # We create the subfolder if necessary
        if not os.path.isdir(f"out/saves/{time_prefix}"):
            os.mkdir(f"out/saves/{time_prefix}")

        # We save the state dict of the model in the saves folder
        torch.save(self.model.state_dict(), f"out/saves/{time_prefix}/{time_prefix}_{epoch+1:03d}.pth")