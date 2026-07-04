#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains a dummy dataloader which can be used to measure the inference time / memory usage
/ FLOPS / number of parameters of a model using data with a specific resolution.
"""

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from transforms.transforms import PadToMaxSize, RandomCropAlignedWithPatches


class DummyDataset(Dataset):
  """
  A dataloader for the dummy dataset.
  """

  def __init__(self, data_resolution: tuple[int, int], evt_channels: int, nb_sequences: int,
               len_sequence: int, transform: Compose | None = None):
    # We save the required resolution of the LiDAR/RGB/event data, as well as the number of channels
    # for the event data
    self.data_resolution = data_resolution
    self.evt_channels = evt_channels

    # We save the number of sequences that should be generated
    self.nb_sequences = nb_sequences

    # We save the length of each sequence, i.e., the number of LiDAR point clouds it contains
    self.len_sequence = len_sequence

    # And we also save the required transform(s)
    self.transform = transform


  def __getitem__(self, _: int) -> list[list[Tensor]]:
    # We create dummy LiDAR/RGB/event/GT data with the required resolution
    dummy_lidar = torch.empty((1, *self.data_resolution))
    dummy_rgb = torch.empty((3, *self.data_resolution))
    dummy_event = torch.empty((self.evt_channels, *self.data_resolution))
    dummy_gt = torch.empty((1, *self.data_resolution))

    # We fill them with normal noise
    nn.init.normal_(dummy_lidar)
    nn.init.normal_(dummy_rgb)
    nn.init.normal_(dummy_event)
    nn.init.normal_(dummy_gt)

    # We initialize the sequence with this data
    sequence = []
    for i in range(self.len_sequence):
      sequence.append([dummy_lidar, dummy_rgb, dummy_event, dummy_gt, dummy_gt])
      sequence.append([None,        None,      dummy_event, dummy_gt, dummy_gt])
      sequence.append([None,        dummy_rgb, dummy_event, dummy_gt, dummy_gt])
      sequence.append([None,        None,      dummy_event, dummy_gt, dummy_gt])
      sequence.append([None,        dummy_rgb, dummy_event, dummy_gt, dummy_gt])
      sequence.append([None,        None,      dummy_event, dummy_gt, dummy_gt])

    # We save the RNG state for the transform operations, as it should be consistent on the whole
    # sequence
    saved_rng_state = torch.get_rng_state()

    # Then, for each item in the sequence (an item is an array containing 1 LiDAR image, 1 RGB
    # image, 1 event volume, 1 "before" depth image, 1 "after" depth image)...
    for item in sequence:
      # For each element in the item...
      for i, elem in enumerate(item):
        # We determine if the data is available
        if elem is None:
          # If not, it is replaced by a Tensor containing a single "nan" value
          item[i] = torch.tensor([float("nan")])
        else:
          # Otherwise we save the image size
          initial_img_size = elem.shape[-2:]

          # And we apply the transform if necessary
          if self.transform is not None:
            torch.set_rng_state(saved_rng_state)
            item[i] = self.transform(elem)

      # We also have to add info about the padding and cropping to the item
      pad_pos = torch.tensor([0, 0, 0, 0], dtype=torch.int)
      crop_pos = torch.tensor([0, 0], dtype=torch.int)
      if self.transform is not None:
        for transform in self.transform.transforms:
          if isinstance(transform, PadToMaxSize):
            top, bottom, left, right = transform.get_params(initial_img_size, transform.size)
            pad_pos = torch.tensor([top, bottom, left, right], dtype=torch.int)
          elif isinstance(transform, RandomCropAlignedWithPatches):
            crop_pos_y, crop_pos_x, _, _ = transform.get_params(initial_img_size,
                                                                transform.out_size,
                                                                transform.patch_size)
            crop_pos = torch.tensor([crop_pos_x, crop_pos_y], dtype=torch.int)
      item.append(pad_pos)
      item.append(crop_pos)

    # And we return the sequence
    return sequence


  def __len__(self) -> int:
    """
    Returns the number of sequences that were found in the given folder
    """
    return self.nb_sequences
