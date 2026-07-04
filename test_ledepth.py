#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file can be used to test the ALED & DELTA networks, on either the SLED, the MVSEC, or the M3ED
datasets, as described in our "DELTA: Dense Depth from Events and LiDAR using Transformer's
Attention" article (CVPRW 2025).
Note: as no parallelism is used here, contrary to the training, this script should be launched
directly, without the use of torchrun. See the README for more details.
"""

import argparse
import json
import os  # Added os for potential path handling

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from dataset.loaders.dummy_dataset_loader import DummyDataset
from dataset.loaders.preprocessed_dataset_loader import PreprocessedDataset
from models.aled import ALED
# from models.ledepth_ssm_unet import LEDepth
from models.lemon_v4 import LEDepth

from models.delta import DELTA

from trainer_tester.tester_ledepth import Tester_LEDepth
from transforms.transforms import PadToMaxSize


def parse_args():
    """Args parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to the JSON config file to use for testing")
    parser.add_argument("checkpoint", help="Path to the .pth checkpoint file to use for testing")
    return parser.parse_args()


def display_count_parameters(model: nn.Module) -> int:
    """
  Utility function to count and display the number of parameters of a network in PyTorch.
  Thanks to https://stackoverflow.com/a/62508086
  """
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        print(name, ":", params)
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    """Main function"""

    # Before doing anything, we must change the torch multiprocessing sharing strategy, to avoid
    # having issues with the shared memory
    torch.multiprocessing.set_sharing_strategy("file_system")

    # We start by loading the config file given by the user
    args = parse_args()
    with open(args.config_file, encoding="utf-8") as cfg_file:
        config = json.load(cfg_file)

    # We configure the device for PyTorch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # We collect the patch size (only for attention-based models)
    if config["model"] == "ALED":
        patch_size = None
    else:
        patch_size = config["patch_size"]

    # We setup the transforms we will perform on the test dataset, i.e. padding (if required)
    if config["transforms"]["pad"]["pad_input"]:
        padded_img_size_x = config["transforms"]["pad"]["padded_image_size_x"]
        padded_img_size_y = config["transforms"]["pad"]["padded_image_size_y"]
        test_transforms = Compose([PadToMaxSize((padded_img_size_y, padded_img_size_x))])
    else:
        test_transforms = None

    # We collect the batch_size and num_workers parameters from the config file
    batch_size = config["batch_size_test"]
    num_workers = config["num_workers"]

    # We load the dataset, which is either:
    # - a dummy dataset if we only want to measure the computational complexity of the model;
    # - or the correct dataset otherwise
    if config["measure_computational_complexity"]:
        # We get the number of channels to use for the event data
        if config["model"] in ("DELTA", "LEDepth", "LEDepth_SSM_UNET"):
            nb_evt_channels = 4
        elif config["model"] == "ALED":
            if config["dataset"]["name"] == "M3ED":
                nb_evt_channels = 4
            else:
                nb_evt_channels = 10
        else:
            nb_evt_channels = 4
            # raise NotImplementedError(f"Model {config['model']} is not implemented")

        # We get the resolution to use
        if config["dataset"]["name"] in ("SLED", "M3ED"):
            data_resolution = (720, 1280)
        elif config["dataset"]["name"] == "DSEC":
            data_resolution = (480, 640)
        elif config["dataset"]["name"] == "MVSEC":
            data_resolution = (260, 346)
        else:
            raise NotImplementedError(f"Dataset {config['dataset']['name']} is not implemented")

        # We initialize the dataset itself
        test_dataset = DummyDataset(data_resolution, nb_evt_channels, 1, 100, test_transforms)
    else:
        # We collect the path to the dataset
        test_dataset_path = config["dataset"]["path_test"]

        # We collect whether the dataset is compressed or not
        test_is_zipped = config["dataset"]["test_is_zipped"]

        # We load the rules to only keep some elements of the training/validation set (because the
        # validation set can be quite large in the case of SLED for instance, and so it might be helpful
        # to only use some of the sequences instead of all of them)
        # Note that if the rule is not set (""), all sequences are accepted
        test_subset_rule = config["dataset"]["test_subset_rule"]

        # We initialize the dataset itself
        test_dataset = PreprocessedDataset(test_dataset_path, test_is_zipped, test_subset_rule,
                                           test_transforms)

    # We create the dataloader (with pin_memory=False, since each sequence is going to be used only
    # once)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, persistent_workers=num_workers > 0,
                                 pin_memory=False)

    # We reset the GPU memory cache, to avoid using unneccessary memory
    # See https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113/2
    torch.cuda.empty_cache()

    # We determine the number of output channels for the model
    out_channels = 2 if config["predict_af_depths"] else 1

    # We initialize the network (based on the model selected in the config file)
    # For some reason, ALED performs very badly on M3ED with 10-channel event data, so we use the
    # 4-channel format used in DELTA instead for this dataset
    if config["model"] == "ALED":
        if config["dataset"]["name"] == "M3ED":
            model = ALED(1, 4, out_channels)
        else:
            model = ALED(1, 10, out_channels)
    elif config["model"] == "DELTA":
        model = DELTA(1, 4, out_channels, 2, patch_size, 1024, 4096, 4, 128)
    elif config["model"] == "LEDepth":
        model = LEDepth(1, 4, out_channels, dimensionality=96,num_heads = 1, patch_size=patch_size)
    elif config["model"] == "LEDepth_SSM_UNET":
        model = LEDepth(
            lidar_chans=1,
            event_chans=4,
            out_chans=out_channels,
            patch_size=patch_size,
            dims=96,
            depths=[2, 2, 9, 2],
            decoder_depths=[2, 2, 2],
            imgsize=224
        )
    else:
        model = LEDepth(
            lidar_chans=1,
            event_chans=4,
            out_chans=out_channels,
            patch_size=patch_size,
            dims=128,
            depths=[2, 2, 18, 2],
            decoder_depths=[2, 2, 2],
            imgsize=224
        )
        # raise NotImplementedError(f"Model {config['model']} is not implemented")

    model.to(device)

    # We display its number of parameters if needed
    if config["measure_computational_complexity"]:
        display_count_parameters(model)

    # We initialize the tester, which is a wrapper class for the testing of the model
    # Note: Tester_LEDepth has been updated to handle the new model calling convention
    tester = Tester_LEDepth(model, test_dataloader, device, config)

    # We give it the checkpoint to load
    tester.load_model_checkpoint(args.checkpoint)

    # And finally, we launch the testing on the whole dataset
    tester.test()


if __name__ == "__main__":
    main()

#python test_ledepth.py configs/LEDepth/test_sled.json /root/autodl-tmp/project/out/saves/20251216_190137/20251216_190137_080.pth
#python test_ledepth.py configs/LEDepth/test_mvsec.json /root/autodl-tmp/project/out/saves/20260308_172124/20260308_172124_100.pth
