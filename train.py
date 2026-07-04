#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file can be used to train or finetune the ALED & DELTA & LEDEPTH variants.
"""

import argparse
from datetime import datetime
import json
import os
from time import sleep
import importlib  # [新增] 用于动态导入

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import Compose
from tqdm import tqdm

from dataset.loaders.preprocessed_dataset_loader import PreprocessedDataset
from losses.losses import L1MSGLoss

from models.aled import ALED
from models.delta import DELTA

# [修改点 1] 移除了所有硬编码的 LEDepth import
# from models.ledepth_ssm_smst import LEDepth
# from models.ledepth_ssm_unet import FusionVMambaUNet

from transforms.transforms import PadToMaxSize, RandomCropAlignedWithPatches
from trainer_tester.trainer_ledepth import Trainer


# torch.autograd.set_detect_anomaly(True)

def parse_args():
    """Args parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to the JSON config file to use for training")
    parser.add_argument("--cp", default=None, help="Checkpoint to restart from (optional)")
    return parser.parse_args()


def display_count_parameters(model: nn.Module) -> int:
    """
  Utility function to count and display the number of parameters of a network in PyTorch.
  """
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        # print(name, ":", params)
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    """
  Main function, used for training and validating the network.
  """

    # We start by initializing the process group, as required by torchrun
    dist.init_process_group(backend="nccl")

    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    nb_gpus = int(os.environ["LOCAL_WORLD_SIZE"])

    # We load the config file given by the user
    args = parse_args()
    with open(args.config_file, encoding="utf-8") as cfg_file:
        config = json.load(cfg_file)

    # Tensorboard setup
    if args.cp is not None:
        time_prefix = os.path.split(args.cp)[-1][:15]
        try:
            start_epoch = int(os.path.split(args.cp)[-1][16:19]) + 1
        except ValueError:
            start_epoch = 0
    else:
        time_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_epoch = 0

    if gpu_id == 0:
        if not os.path.isdir("/root/tf-logs"):
            os.makedirs("/root/tf-logs", exist_ok=True)
        writer = SummaryWriter(os.path.join("/root/tf-logs", time_prefix))
    else:
        writer = None

    # Patch size collection
    # [修改点 2] 只要不是 ALED，都尝试读取 patch_size，或者你可以根据动态逻辑判断
    if config["model"] != "ALED":
        patch_size = config.get("patch_size", None)
    else:
        patch_size = None

    # Transforms setup
    train_transforms_list = []
    if config["transforms"]["pad"]["pad_input"]:
        padded_img_size_x = config["transforms"]["pad"]["padded_image_size_x"]
        padded_img_size_y = config["transforms"]["pad"]["padded_image_size_y"]
        train_transforms_list.append(PadToMaxSize((padded_img_size_y, padded_img_size_x)))
    if config["transforms"]["crop"]["crop_input"]:
        crop_size = config["transforms"]["crop"]["crop_size"]
        train_transforms_list.append(RandomCropAlignedWithPatches(crop_size, patch_size))
    train_transforms = Compose(train_transforms_list)

    if config["transforms"]["pad"]["pad_input"]:
        val_transforms = Compose([PadToMaxSize((padded_img_size_y, padded_img_size_x))])
    else:
        val_transforms = None

    batch_size_train = config["batch_size_train"]
    batch_size_train_per_gpu = max(1, batch_size_train // nb_gpus)
    num_workers = config["num_workers"]
    batch_size_val = 1

    train_is_zipped = config["dataset"]["train_is_zipped"]
    val_is_zipped = config["dataset"]["val_is_zipped"]
    train_subset_rule = config["dataset"]["train_subset_rule"]
    val_subset_rule = config["dataset"]["val_subset_rule"]

    # Dataset Loading
    train_dataset_path = config["dataset"]["path_train"]
    if not os.path.exists(train_dataset_path):
        raise FileNotFoundError(f"train dataset not exist: {train_dataset_path}")
    if gpu_id == 0:
        print(f"[GPU 0] train_dataset: {train_dataset_path}")

    train_dataset = PreprocessedDataset(train_dataset_path, train_is_zipped, train_subset_rule,
                                        train_transforms)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size_train_per_gpu,
                                  shuffle=False, sampler=train_sampler, num_workers=num_workers,
                                  persistent_workers=num_workers > 0, pin_memory=True)

    if gpu_id == 0:
        val_dataset_path = config["dataset"]["path_val"]
        if not os.path.exists(val_dataset_path):
            raise FileNotFoundError(f"val dataset not exist: {val_dataset_path}")
        print(f"[GPU 0] val_dataset: {val_dataset_path}")
        val_dataset = PreprocessedDataset(val_dataset_path, val_is_zipped, val_subset_rule,
                                          val_transforms)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size_val, shuffle=False,
                                    num_workers=num_workers, persistent_workers=num_workers > 0,
                                    pin_memory=True)
    else:
        val_dataloader = None

    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()

    out_channels = 2 if config["predict_af_depths"] else 1

    # ============================================================================================
    # [修改点 3] 动态导入模型逻辑
    # ============================================================================================

    # 1. 优先处理旧的硬编码模型 (ALED, DELTA) - 保持兼容性
    if config["model"] == "ALED":
        if config["dataset"]["name"] == "M3ED":
            model = ALED(1, 4, out_channels)
        else:
            model = ALED(1, 10, out_channels)

    elif config["model"] == "DELTA":
        model = DELTA(1, 4, out_channels, 2, patch_size, 1024, 4096, 4, 128)

    else:
        # 2. 通用动态加载逻辑 (适用于 LEDEPTH_SSM_UNET, LEDepth 等)
        try:
            # 获取模型名称并转小写作为文件名
            model_name = config["model"]  # 例如: "LEDEPTH_SSM_UNET"
            module_name = model_name.lower()  # 例如: "ledepth_ssm_unet"

            # 动态导入模块: models.ledepth_ssm_unet
            print(f"[GPU {gpu_id}] Loading module: models.{module_name}")
            module = importlib.import_module(f"models.{module_name}")

            ModelClass = getattr(module, "LEDepth")
            model = ModelClass(
                lidar_chans=1,
                event_chans=4,
                out_chans=out_channels,
                patch_size=patch_size,
                dims=128,  # Tiny Config
                depths=[2, 2, 18, 2],
                decoder_depths=[2, 2, 2],
                imgsize=224
            )

        except ImportError as e:
            raise ImportError(f"无法导入模块 models.{module_name}。请检查文件名是否为 {module_name}.py。错误信息: {e}")
        except AttributeError:
            raise AttributeError(
                f"在模块 models.{module_name} 中找不到类 'LEDepth'。请确保类名为 LEDepth 或在文件中添加了别名 (LEDepth = YourClassName)。")
        except Exception as e:
            raise RuntimeError(f"初始化模型 {model_name} 时发生错误: {e}")

    # SyncBN & DDP
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    if config["model"] != "ALED":
        model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
    else:
        model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=False)  # ALED 通常不需要

    if gpu_id == 0:
        display_count_parameters(model)

    # Epochs
    num_epochs = config["epochs"]

    # LR Scheduler
    initial_learning_rate = config["initial_learning_rate"]
    final_learning_rate = config["final_learning_rate"]
    if final_learning_rate != initial_learning_rate:
        lr_lambda = lambda epoch: (final_learning_rate / initial_learning_rate) ** (epoch / (num_epochs - 1))
    else:
        lr_lambda = lambda _: 1.0

    criterion = L1MSGLoss(5)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if start_epoch != 0:
        if gpu_id == 0:
            print("!! Fast-forwarding scheduler to epoch", start_epoch)
        for _ in range(start_epoch):
            scheduler.step()

    # Trainer
    trainer = Trainer(model, train_dataloader, val_dataloader, criterion, optimizer, writer, config)

    if args.cp is not None:
        trainer.load_model_checkpoint(args.cp)

    # Loop
    for epoch in tqdm(range(start_epoch, num_epochs), "Epochs", disable=gpu_id != 0):
        train_sampler.set_epoch(epoch)
        trainer.train(epoch)

        if nb_gpus > 1:
            dist.barrier()

        if gpu_id == 0:
            trainer.val(epoch)
            trainer.save_model_checkpoint(time_prefix, epoch)

        scheduler.step()

        if nb_gpus > 1:
            dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# OMP_NUM_THREADS=8 MASTER_ADDR="localhost" torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py configs/LEDepth/train_sled.json