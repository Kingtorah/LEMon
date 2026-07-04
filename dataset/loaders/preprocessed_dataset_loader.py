#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains a dataloader which can be used to load any of the preprocessed datasets.
"""

from glob import glob
from io import BytesIO
from os import path, sep
import zipfile

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from transforms.transforms import PadToMaxSize, RandomCropAlignedWithPatches


class PreprocessedDataset(Dataset):
  """
  A dataloader for the preprocessed SLED, MVSEC, and M3ED datasets.
  To use this dataloader, the dataset must have already been preprocessed using the corresponding
  script (using dataset/preprocess/preprocess_[...]_dataset.py, see the README for more info).
  """

  def __init__(self, path_to_dataset: str, is_dataset_zipped: bool, dataset_subset_rule: str,
               transform: Compose | None = None):
    # We check that the path points to a folder
    if not path.isdir(path_to_dataset):
      raise FileNotFoundError("The path to the dataset should be a folder")

    # We collect the list of all the compressed .pt.zip or uncompressed .pt files in the folder
    if is_dataset_zipped:
      file_extension = ".pt.zip"
    else:
      file_extension = ".pt"
    self.sequences_paths = sorted(glob(f"{path_to_dataset}/*{file_extension}"))

    # If the folder doesn't contain at least one preprocessed file, we throw an exception
    if not self.sequences_paths:
      raise FileNotFoundError(f"The given folder ({path_to_dataset}) doesn't contain any "
                              f"{file_extension} file!")

    # If required, we subsample the dataset
    if dataset_subset_rule != "":
      subsampled_sequences_paths = []

      for idx, sequence_path in enumerate(self.sequences_paths):
        if eval(dataset_subset_rule):
          subsampled_sequences_paths.append(sequence_path)

      self.sequences_paths = subsampled_sequences_paths

    # We also save whether the dataset is zipped or not, and the required transform(s)
    self.is_dataset_zipped = is_dataset_zipped
    self.transform = transform


  def __getitem__(self, index: int) -> list[list[Tensor]]:
    # As the dataset has already been preprocessed, we only have to load the correct .pt file
    # For that purpose, if the dataset is zipped, we must first open the zipfile, read the single
    # .pt compressed file in it, and then extract its content and load it with PyTorch
    # If the dataset is not zipped, then we just have to load the .pt file
    if self.is_dataset_zipped:
      try:
          with zipfile.ZipFile(self.sequences_paths[index], "r") as zip_file:
            compressed_file_name = self.sequences_paths[index].split(sep)[-1][:-4]
            with zip_file.open(compressed_file_name) as compressed_file:
              sequence_buffer = compressed_file.read()
          sequence = torch.load(BytesIO(sequence_buffer), weights_only=True)
      except zipfile.BadZipFile as e:
        print(f"文件损坏: {self.sequences_paths[index]}, 错误: {e}")
        raise
    else:
      sequence = torch.load(self.sequences_paths[index], weights_only=True)

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
    return len(self.sequences_paths)
