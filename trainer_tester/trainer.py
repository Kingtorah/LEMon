#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains a trainer class, which can be used to train, finetune, and validate the ALED &
DELTA networks, on either the SLED, the MVSEC, or the M3ED datasets, as described in our
"DELTA: Dense Depth from Events and LiDAR using Transformer's Attention" article (CVPRW 2025).
"""

import os
import sys  # [Added] 用于控制标准输出
import datetime  # [Added] 用于生成日志时间戳

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from metrics.metrics import l1_error
from visualization.visualization import depth_image_to_img, event_volume_to_img, lidar_proj_to_img


# [Added] 一个辅助类，用于将输出同时写到终端和文件
class TeeLogger(object):
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        # 写入终端
        self.terminal.write(message)
        # 写入文件
        self.log.write(message)
        # 强制刷新缓冲区，确保实时写入（防止崩溃时丢失最后几行）
        self.log.flush()
        self.terminal.flush()

    def flush(self):
        # 满足 python 接口要求，某些库（如 tqdm）需要 flush 方法
        self.terminal.flush()
        self.log.flush()


class Trainer():
    """
  A trainer for the ALED & DELTA & LEDepth networks
  """

    def __init__(self, model: DistributedDataParallel, train_dataloader: DataLoader,
                 val_dataloader: DataLoader | None, loss_criterion: nn.Module, optimizer: Optimizer,
                 tensorboard_writer: SummaryWriter | None, config: dict):

        # We identify the GPU/device in use FIRST to setup logging immediately
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.device = f"cuda:{self.gpu_id}"

        # [Added] 日志记录模块配置 (仅在 GPU 0 上执行，避免多进程写入冲突)
        if self.gpu_id == 0:
            # 创建日志目录
            log_dir = "out/logs"
            if not os.path.isdir("out"):
                os.mkdir("out")
            if not os.path.isdir(log_dir):
                os.mkdir(log_dir)

            # 生成带时间戳的日志文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_log = config.get("model", "Model")
            log_filename = os.path.join(log_dir, f"train_log_{model_name_log}_{timestamp}.txt")

            print(f"--> [System] Terminal output is being mirrored to: {log_filename}")

            # 重定向 stdout (print) 和 stderr (tqdm/errors)
            sys.stdout = TeeLogger(log_filename, sys.stdout)
            sys.stderr = TeeLogger(log_filename, sys.stderr)

        # We set the model, the number of epochs, the dataloaders, the loss criterion, the optimizer,
        # the Tensorboard writer, and the config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = loss_criterion
        self.optimizer = optimizer
        self.writer = tensorboard_writer

        # We also initialize the scaler (due to the use of AMP)
        # self.scaler = torch.GradScaler("cuda")
        self.scaler = torch.cuda.amp.GradScaler()

        # We save the name of the model
        self.model_name = config["model"]

        # We collect the total number of epochs
        self.num_epochs = config["epochs"]

        # We collect the patch size (only for attention-based models)
        if self.model_name in ("DELTA", "LEDepth"):
            self.patch_size = config["patch_size"]
        else:
            self.patch_size = None

        # We determine the number of output channels of the model
        self.out_channels = 2 if config["predict_af_depths"] else 1

        # We save the weight for the MSG loss for the first epoch
        self.weight_msg_epoch_0 = config["weight_msg_epoch_0"]

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
        running_loss = 0.0
        running_len = 0

        # We set the weights for the loss
        weight_l1 = 1.0
        if epoch == 0:
            weight_ms = self.weight_msg_epoch_0
        else:
            weight_ms = 1.0

        # For each sequence extracted from the dataset...
        # [Note] tqdm writes to stderr, which is now redirected to our log file as well
        for seq_idx, sequence in enumerate(tqdm(self.train_dataloader, "Training", leave=False,
                                                disable=self.gpu_id != 0)):
            # We set to zero the losses for the sequence (used for display in Tensorboard)
            seq_bf_loss_l1 = 0.0
            seq_bf_loss_ms = 0.0
            seq_af_loss_l1 = 0.0
            seq_af_loss_ms = 0.0
            seq_loss = 0.0
            seq_len = 0

            # We initialize the memories
            central_mem = None
            prop_mem = None

            # We initialize the loss for the sequence
            loss = 0

            # For each item (1 LiDAR image, 1 RGB image, 1 event volume, 1 "before" depth image,
            # 1 "after" depth image, 1 padding info, 1 cropping info) in the sequence...
            for item_idx, item in enumerate(sequence):
                # print("train is here--------------------------------")
                # We extract the data from the sequence, we check if they are available, and we upload
                # them to the device if it is the case
                # We also reset the LiDAR prop mem if LiDAR data is available, and we also make sure that
                # the ground truth depths are in the range [0, 1]
                lidar_proj, rgb_image, event_volume, bf_depths, af_depths, _, crop_positions = item

                lidar_proj_available = not torch.all(torch.isnan(lidar_proj))
                rgb_image_available = not torch.all(torch.isnan(rgb_image))
                event_volume_available = not torch.all(torch.isnan(event_volume))
                bf_depths_available = not torch.all(torch.isnan(bf_depths))
                af_depths_available = not torch.all(torch.isnan(af_depths))

                if lidar_proj_available:
                    lidar_proj = lidar_proj.to(self.device, non_blocking=True)
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
                with torch.autocast("cuda"):
                    # ==================================================================
                    # Debugging Block
                    debug_targets = {
                        "Input_LiDAR": lidar_proj,
                        "Input_Event": event_volume,
                        "GT_BF_Depths": bf_depths,
                        "GT_AF_Depths": af_depths,
                        "Mem_Central": central_mem,
                        "Mem_Prop": prop_mem
                    }

                    found_nan = False
                    crash_seq = f"epoch{epoch}_seq{seq_idx}"

                    for name, data in debug_targets.items():
                        if data is None: continue

                        # 情况 A: 数据是 List
                        if isinstance(data, list):
                            for i, item in enumerate(data):
                                if torch.isnan(item).any() or torch.isinf(item).any():
                                    # This print will now go to the log file as well
                                    print(f"💀 [CRITICAL] {name} (Layer {i}) 坏了！位置: {crash_seq}")
                                    torch.save(item, f"crash_{name}_layer{i}_{crash_seq}.pt")
                                    found_nan = True

                        # 情况 B: 数据是 Tensor
                        elif torch.isnan(data).any() or torch.isinf(data).any():
                            # This print will now go to the log file as well
                            print(f"💀 [CRITICAL] {name} 坏了！位置: {crash_seq}")
                            torch.save(data, f"crash_{name}_{crash_seq}.pt")
                            found_nan = True

                    if found_nan:
                        print("!!! 检测到 NaN，已保存 .pt 文件，训练终止以保护现场 !!!")
                        # We explicitly flush before raising error to ensure log file has the crash info
                        sys.stdout.flush()
                        sys.stderr.flush()
                        raise ValueError(f"NaN Detected at {crash_seq}")
                    # ==================================================================

                    # We feed the data to the network, to derive a prediction and get the updated memories
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
                    # ==================================================================
                    if torch.isnan(pred_depths).any():
                        print(
                            f"!!! [GPU {self.gpu_id}] Model Output (pred_depths) is NaN at Epoch {epoch} Seq {seq_idx} !!!")
                        torch.save(lidar_proj, f"nan_input_lidar_{seq_idx}.pt")
                        torch.save(event_volume, f"nan_input_event_volume_{seq_idx}.pt")
                        sys.stdout.flush()
                        raise ValueError("Model output is NaN")
                    # ==================================================================
                    # After every prediction, we compute the losses
                    if bf_depths_available:
                        bf_loss_l1, bf_loss_ms = self.criterion(pred_depths[:, [0], :, :], bf_depths)
                    else:
                        bf_loss_l1, bf_loss_ms = torch.tensor(0.0), torch.tensor(0.0)

                    if self.out_channels == 2 and af_depths_available:
                        af_loss_l1, af_loss_ms = self.criterion(pred_depths[:, [1], :, :], af_depths)
                    else:
                        af_loss_l1, af_loss_ms = torch.tensor(0.0), torch.tensor(0.0)

                    # And finally, the complete loss is the weighted sum of all these losses
                    loss_ = weight_l1 * (bf_loss_l1 + af_loss_l1) + \
                            weight_ms * (bf_loss_ms + af_loss_ms)
                    loss += loss_

                    # We save the losses for analysis (only on the first GPU)
                    if self.gpu_id == 0:
                        seq_bf_loss_l1 += bf_loss_l1.item()
                        seq_bf_loss_ms += bf_loss_ms.item()
                        seq_af_loss_l1 += af_loss_l1.item()
                        seq_af_loss_ms += af_loss_ms.item()
                        seq_loss += loss_.item()
                        seq_len += 1
                        running_bf_loss_l1 += bf_loss_l1.item()
                        running_bf_loss_ms += bf_loss_ms.item()
                        running_af_loss_l1 += af_loss_l1.item()
                        running_af_loss_ms += af_loss_ms.item()
                        running_loss += loss_.item()
                        running_len += 1

                    # Once all the events associated with the LiDAR cloud have been processed, we compute the
                    # gradients
                    if (item_idx + 1) % self.accumulation_steps_train == 0:
                        # Note: we leave the mixed precision mode here
                        with torch.autocast("cuda", enabled=False):
                            # We apply the backwards pass
                            self.optimizer.zero_grad()
                            self.scaler.scale(loss).backward()

                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                            self.scaler.step(self.optimizer)
                            self.scaler.update()

                        # And we don't forget to detach the central memory/memories, as well as to reset the
                        # loss
                        if isinstance(central_mem, list):
                            central_mem = [mem.detach() for mem in central_mem]
                        else:
                            central_mem = central_mem.detach()
                        loss = 0

            # Finally, we process the display of the loss (only on the first GPU)
            if self.gpu_id == 0:
                # We compute the current index
                curr_idx = epoch * self.total_nb_seq_train + seq_idx

                # We write the losses in tensorboard
                self.writer.add_scalar("Loss on bf depths (L1)", seq_bf_loss_l1 / seq_len, curr_idx)
                self.writer.add_scalar("Loss on bf depths (MS)", seq_bf_loss_ms / seq_len, curr_idx)
                self.writer.add_scalar("Loss on af depths (L1)", seq_af_loss_l1 / seq_len, curr_idx)
                self.writer.add_scalar("Loss on af depths (MS)", seq_af_loss_ms / seq_len, curr_idx)
                self.writer.add_scalar("Total loss", seq_loss / seq_len, curr_idx)

                # And, if it is needed, we display them
                if (seq_idx + 1) % self.losses_display_every_x == 0:
                    # tqdm.write automatically handles writing to the correct stream, which we redirected
                    tqdm.write(f"Epoch {epoch + 1} / {self.num_epochs}, "
                               f"seq. {seq_idx + 1}/{self.total_nb_seq_train}, "
                               f"loss bf l1 = {running_bf_loss_l1 / running_len:.5f}, "
                               f"loss bf ms = {running_bf_loss_ms / running_len:.5f}, "
                               f"loss af l1 = {running_af_loss_l1 / running_len:.5f}, "
                               f"loss af ms = {running_af_loss_ms / running_len:.5f}, "
                               f"total loss = {running_loss / running_len:.5f}")
                    running_bf_loss_l1 = 0.0
                    running_bf_loss_ms = 0.0
                    running_af_loss_l1 = 0.0
                    running_af_loss_ms = 0.0
                    running_loss = 0.0
                    running_len = 0

    @torch.inference_mode()
    def val(self, epoch: int) -> None:
        """
    Run validation of the model.
    The code run here is nearly the same as the training one (only parts added are the images being
    pushed to Tensorboard and the final metric evaluation part), hence the few comments
    """

        # To start, we must not forget to set the model to evaluation mode
        self.model.eval()
        # The criterion for validation is a simple L1 error
        # More complete evaluation metrics are available as part of the testing code
        running_val_bf_error = 0.0
        running_val_af_error = 0.0

        nb_bf_errors = 0
        nb_af_errors = 0

        for seq_idx, sequence in enumerate(tqdm(self.val_dataloader, "Validation", leave=False)):
            total_items = len(sequence)
            central_mem = None
            prop_mem = None

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

                # We compute the coordinates to use to remove the padding
                pad_top, pad_bottom, pad_left, pad_right = pad_positions[0, :]
                min_x = pad_left
                max_x = padded_img_size[1] - pad_right
                min_y = pad_top
                max_y = padded_img_size[0] - pad_bottom

                with torch.autocast("cuda"):
                    if self.model_name == "ALED":
                        pred_depths, central_mem = self.model.module(lidar_proj, event_volume, central_mem)
                    elif self.model_name == "DELTA":
                        pred_depths, central_mem, prop_mem = self.model.module(lidar_proj, event_volume,
                                                                               central_mem, prop_mem,
                                                                               crop_positions)
                    elif self.model_name == "LEDepth":
                        pred_depths, central_mem, prop_mem = self.model.module(lidar_proj, event_volume,
                                                                               central_mem, prop_mem,
                                                                               crop_positions)
                    else:
                        raise NotImplementedError(f"Model {self.model_name} is not implemented")

                # We correct the prediction, to force it to be in the [0, 1] range
                pred_depths[pred_depths < 0.0] = 0.0
                pred_depths[pred_depths > 1.0] = 1.0

                # We remove any padding before using the data further on
                if bf_depths_available:
                    unpadded_bf_depths = bf_depths[:, :, min_y:max_y, min_x:max_x]
                if af_depths_available:
                    unpadded_af_depths = af_depths[:, :, min_y:max_y, min_x:max_x]
                unpadded_pred_depths = pred_depths[:, :, min_y:max_y, min_x:max_x]

                # We save images of the input and output data
                # We begin by computing the current index
                idx = epoch * self.total_nb_seq_val * total_items + seq_idx * total_items + item_idx

                # Display of the LiDAR projection
                if lidar_proj_available:
                    lidar_img = lidar_proj_to_img(lidar_proj[:, :, min_y:max_y, min_x:max_x])
                    self.writer.add_images("LiDAR", lidar_img, idx)

                # Display of the RGB image
                if rgb_image_available:
                    self.writer.add_images("RGB", rgb_image, idx)

                # Display of the event volume
                if event_volume_available:
                    events_img = event_volume_to_img(event_volume[:, :, min_y:max_y, min_x:max_x])
                    self.writer.add_images("Events", events_img, idx)

                # Display of the D_bf depth image
                if bf_depths_available:
                    bf_depth_image_img = depth_image_to_img(unpadded_bf_depths)
                    self.writer.add_images("BF GT depth image", bf_depth_image_img, idx)

                # Display of the D_af depth image
                if af_depths_available:
                    af_depth_image_img = depth_image_to_img(unpadded_af_depths)
                    self.writer.add_images("AF GT depth image", af_depth_image_img, idx)

                # Display of the estimated D_bf depths
                pred_bf_img = depth_image_to_img(unpadded_pred_depths[:, [0], :, :])
                self.writer.add_images("BF Pred", pred_bf_img, idx)

                # Display of the estimated D_af depths
                if self.out_channels == 2:
                    pred_af_img = depth_image_to_img(unpadded_pred_depths[:, [1], :, :])
                    self.writer.add_images("AF Pred", pred_af_img, idx)

                if bf_depths_available:
                    not_nan_mask_bf = ~torch.isnan(unpadded_bf_depths)
                    masked_unpadded_pred_bf = unpadded_pred_depths[:, [0], :, :][not_nan_mask_bf]
                    masked_unpadded_bf_depths = unpadded_bf_depths[not_nan_mask_bf]

                if self.out_channels == 2 and af_depths_available:
                    not_nan_mask_af = ~torch.isnan(unpadded_af_depths)
                    masked_unpadded_pred_af = unpadded_pred_depths[:, [1], :, :][not_nan_mask_af]
                    masked_unpadded_af_depths = unpadded_af_depths[not_nan_mask_af]

                if bf_depths_available:
                    running_val_bf_error += l1_error(masked_unpadded_pred_bf, masked_unpadded_bf_depths)
                    nb_bf_errors += 1

                if self.out_channels == 2 and af_depths_available:
                    running_val_af_error += l1_error(masked_unpadded_pred_af, masked_unpadded_af_depths)
                    nb_af_errors += 1

        # At the end, we save the error on the validation set for analysis
        if nb_bf_errors != 0:
            running_val_bf_error /= nb_bf_errors
        else:
            running_val_bf_error = 0

        if nb_af_errors != 0:
            running_val_af_error /= nb_af_errors
        else:
            running_val_af_error = 0

        self.writer.add_scalar("Val. error on D_bf depths", running_val_bf_error, epoch)
        self.writer.add_scalar("Val. error on D_af depths", running_val_af_error, epoch)
        self.writer.add_scalar("Total val. error", (running_val_bf_error + running_val_af_error) / 2, epoch)

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
        if epoch % save_interval != 0:
            return
        # We create the "saves" folder if necessary
        if not os.path.isdir("out/saves"):
            os.mkdir("out/saves")

        # We create the subfolder if necessary
        if not os.path.isdir(f"out/saves/{time_prefix}"):
            os.mkdir(f"out/saves/{time_prefix}")

        # We save the state dict of the model in the saves folder
        torch.save(self.model.state_dict(), f"out/saves/{time_prefix}/{time_prefix}_{epoch:03d}.pth")