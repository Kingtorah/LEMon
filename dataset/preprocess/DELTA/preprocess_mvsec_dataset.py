#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script can be called to preprocess the MVSEC dataset for the DELTA model, transforming the raw
.hdf5 recordings into PyTorch .pt files, with all the preprocessing steps (data formatting,
normalization, LiDAR projection, ...) already applied.
Preprocessing the dataset has its pros and cons:
- it normalizes the datasets, so that they can all be loaded using a single dataloader;
- it reduces greatly the computational power necessary to load the dataset during training /
  validation / testing;
- it greatly increases disk space usage, as the dataset is converted into multiple small sequences;
- if the preprocessing steps are modified, the whole dataset has to be preprocessed again.
"""

from argparse import ArgumentParser, Namespace
from glob import glob
from io import BytesIO
from os import mkdir, path
import zipfile

import h5py
import numpy as np
from numpy import ndarray
import pandas as pd
import torch
from torch import Tensor
from tqdm.contrib.concurrent import process_map
import yaml


LIDAR_MAX_RANGE = 100
EVTS_BINS = 4


# We define args as a global variable, so that it can be seen by all parallel processes
args = Namespace()


def parse_args():
  """Args parser"""
  parser = ArgumentParser()
  parser.add_argument("set", help="Should be 'train', 'val', or 'test'")
  parser.add_argument("path_raw", help="Path to the folder containing the raw dataset")
  parser.add_argument("path_processed", help="Path to the folder where the dataset will be stored "
    "after preprocessing")
  parser.add_argument("-z", "--zipped", action="store_true", help="If this flag is set, the "
    "preprocessed sequences will be compressed as .pt.zip files (otherwise, they are saved as "
    "regular .pt files)")
  parser.add_argument("-j", type=int, default=1, help="Number of parallel processes spawned to "
    "preprocess the dataset. If not specified, a single process is spawned")
  return parser.parse_args()


def compute_lidar_projection(lidar_cloud: ndarray, lidar_max_range: int, davis_intrinsics: ndarray,
                             davis_dist: ndarray, t_davis_lidar: ndarray) -> Tensor:
  """
  Creates a projection of the point cloud in a 1-channel Tensor.
  Each pixel contains its depth value, normalized between 0 and 1 (pixels with no LiDAR point are
  given a value of 0).
  """

  # We use the calibration parameters to initialize the camera matrix
  K = np.array([[davis_intrinsics[0], 0, davis_intrinsics[2]],
                [0, davis_intrinsics[1], davis_intrinsics[3]],
                [0, 0, 1]])

  # We then filter the point cloud, to only retain points in front of the camera
  lidar_cloud_filt = lidar_cloud[lidar_cloud[:, 0] > 0]

  # We tranform the coordinates to the camera's frame
  pcl_camera_frame = ((t_davis_lidar @ lidar_cloud_filt.T).T)[:, :3]
  pcl_camera_frame_filt = pcl_camera_frame[pcl_camera_frame[:, 2] >= 1.0]
  depths = pcl_camera_frame_filt[:, 2].copy()
  pcl_camera_frame_filt[:, 0] /= depths
  pcl_camera_frame_filt[:, 1] /= depths
  pcl_camera_frame_filt[:, 2] /= depths

  # We apply the distortions
  # See eq. (3) to (8) of the "AprilCal: Assisted and repeatable camera calibration" article from
  # Richardson et al. for more details on the formulas
  pcl_camera_frame_dist = np.ones_like(pcl_camera_frame_filt)
  x = pcl_camera_frame_filt[:, 0]
  y = pcl_camera_frame_filt[:, 1]
  z = pcl_camera_frame_filt[:, 2]
  r_2 = x*x + y*y
  r = np.sqrt(r_2)
  theta = np.arctan2(r, z)
  theta2 = theta*theta
  theta3 = theta2*theta
  theta4 = theta2*theta2
  theta5 = theta4*theta
  theta6 = theta3*theta3
  theta7 = theta6*theta
  theta8 = theta4*theta4
  theta9 = theta8*theta
  theta_d = theta + davis_dist[0]*theta3 + davis_dist[1]*theta5 + davis_dist[2]*theta7 + davis_dist[3]*theta9
  psi = np.arctan2(y, x)
  pcl_camera_frame_dist[:, 0] = theta_d * np.cos(psi)
  pcl_camera_frame_dist[:, 1] = theta_d * np.sin(psi)

  # We project them in the image
  pcl_camera = (K @ pcl_camera_frame_dist.T).T

  # We create the projection, and add each projected LiDAR point to it
  # The projection is composed of 1 channel, containing the normalized depth of each point
  lidar_proj = torch.zeros(1, 260, 346)
  for i, pt in enumerate(pcl_camera[:, :2]):
    if pt[0] >= 0 and pt[0] < 346 and pt[1] >= 0 and pt[1] < 260:
      lidar_proj[0, int(pt[1]), int(pt[0])] = min(depths[i]/lidar_max_range, 1.0)

  # We return the projection
  return lidar_proj


def compute_event_volume(events: ndarray, bins: int) -> Tensor:
  """
  From a numpy array of events, computes an event volume, as described in the "Unsupervised
  Event-based Learning of Optical Flow, Depth, and Egomotion" article by Zhu et al.
  This implementation is optimized for fast computation (which is still a bit slow :c), thanks to
  https://stackoverflow.com/a/55739936
  """

  # We create an empty event volume
  event_volume = np.zeros((bins, 260, 346), np.float32)

  # We compute the t_star value for each event
  t_star = (bins-1)*(events[:, 2]-events[0, 2])/(events[-1, 2]-events[0, 2])

  # We create an index of unique (x, y) events
  evts_x_y = [tuple(e.astype(int)) for e in events[:, [0, 1]]]
  idx, u_evts = pd.factorize(evts_x_y)
  u_evts = np.array([np.array(u_evt) for u_evt in u_evts])

  # Then, for each bin...
  for i in range(bins):
    # We compute the sum of the pol*max(0, 1-abs(bin-t_star)) for each pixel
    sums = np.bincount(idx, events[:, 3]*np.fmax(0, 1-abs(i-t_star)))

    # We set these values inside the event volume
    event_volume[i, u_evts[:, 1], u_evts[:, 0]] = sums

  # We finally return the event volume, in the PyTorch format
  return torch.from_numpy(event_volume)


def compute_depth_image(depth_image_raw: ndarray, lidar_max_range: int) -> Tensor:
  """
  From a raw depth image, computes its equivalent Tensor-based representation
  """

  # We normalize the values based on the max range of the LiDAR
  # Note that the depth image contains values > than 1.0, which should probably be filtered out
  # during training
  depth_image_raw /= lidar_max_range

  # Finally, we transform the numpy matrix to a PyTorch Tensor
  depth_image = torch.from_numpy(depth_image_raw)

  # And we return it
  return depth_image


def sorted_slice_mask(arr: ndarray, l_val: float, r_val: float) -> ndarray:
  """
  Returns a mask of array arr (arr should be sorted!) such that values v of arr verifying l < v <= r
  are set to True in the mask (and the others are set to False)
  """

  start = np.searchsorted(arr, l_val, "right")
  end = np.searchsorted(arr, r_val, "left")
  out_arr = np.full_like(arr, False, dtype=bool)
  out_arr[start:end+1] = True
  return out_arr


def preprocess_recording(paths: list[str]) -> None:
  """
  Recording preprocessing function, which can be called in parallel on all the recordings.
  """

  # We get the prefix of the file (outdoor_day1, ...)
  prefix = paths[0].split('/')[-1][:-10]

  # We open and read data from the _data and _gt files
  data_recording = h5py.File(paths[0])
  gt_recording = h5py.File(paths[1])
  events = data_recording["davis"]["left"]["events"][:]
  events_ts = events[:, 2]
  lidar_clouds = data_recording["velodyne"]["scans"][:]
  lidar_clouds_ts = data_recording["velodyne"]["scans_ts"][:]
  depths = gt_recording["davis"]["left"]["depth_image_raw"][:]
  depths_ts = gt_recording["davis"]["left"]["depth_image_raw_ts"][:]

  # We open and read data from the calibration file
  with zipfile.ZipFile(paths[2]) as calib_zip_file:
    calib_file_name = ""
    for file in calib_zip_file.namelist():
      if file.endswith(".yaml"):
        calib_file_name = file
    if not calib_file_name:
      raise FileNotFoundError(f"Could not find a .yaml file in {paths[2]}")
    with calib_zip_file.open(calib_file_name) as stream:
      yaml_data = yaml.safe_load(stream)
      davis_intrinsics = np.array(yaml_data["cam0"]["intrinsics"])
      davis_dist = np.array(yaml_data["cam0"]["distortion_coeffs"])
      t_davis_lidar = np.array(yaml_data["T_cam0_lidar"])

  # We have to filter the MVSEC dataset before using it: LiDAR scans are always available, but
  # events and ground truth depth images are not necessarily available at the beginning or at the
  # end of each sequence. So, based on the timestamps of the LiDAR clouds, we only keep those for
  # which both a D_bf and a D_af depth images + an event volume can be associated
  valid_data = []
  for i in range(lidar_clouds.shape[0]-1):
    # We create the array that will contain the LiDAR cloud, the D_bf and D_af depth images, an the
    # events, in which we already place the LiDAR cloud
    possible_valid_data = [lidar_clouds[i, :, :]]

    # To know which events and depth images should be extracted, we set the start timestamp as the
    # one of the current LiDAR cloud, and set the end timestamp as the timestamp of the next LiDAR
    # cloud
    start_ts = lidar_clouds_ts[i]
    end_ts = lidar_clouds_ts[i+1]

    # We check if we can get a bf_depth_image in this interval of time
    bf_depth_image_ts_mask = np.bitwise_and(depths_ts >= (start_ts - 0.001),
                                            depths_ts <= (start_ts + 0.001))
    if np.sum(bf_depth_image_ts_mask) == 0:
      continue
    possible_valid_data.append(depths[bf_depth_image_ts_mask])

    # We do the same for the af_depth_image
    af_depth_images_ts_mask = np.bitwise_and(depths_ts >= (end_ts - 0.001),
                                             depths_ts <= (end_ts + 0.001))
    if np.sum(af_depth_images_ts_mask) == 0:
      continue
    possible_valid_data.append(depths[af_depth_images_ts_mask])

    # And we finish with the events
    events_ts_mask = sorted_slice_mask(events_ts, start_ts, end_ts)
    if np.sum(events_ts_mask) == 0:
      continue
    possible_valid_data.append(events[events_ts_mask])

    # If we get here: congrats, we were able to get all the required data! So, we add it to the
    # valid_data array
    valid_data.append(possible_valid_data)

  # The number of LiDAR images per sequence is determined based on the set in use:
  # - 3 for the train set
  # - 20 for the validation set
  # - all for the test set (we keep the full recording, without any splitting)
  if args.set == "train":
    lidar_clouds_per_seq = 3
  elif args.set == "val":
    lidar_clouds_per_seq = 20
  else:
    lidar_clouds_per_seq = len(valid_data)

  # Each recording allows us to generate N/L sequences, where N is the total of valid data that was
  # previously filtered, and L is the total number of successive point clouds in a sequence
  nb_seq = len(valid_data) // lidar_clouds_per_seq

  # We generate each of the nb_seq sequences
  for i in range(nb_seq):
    sequence = []

    # Each sequence contains L successive LiDAR clouds from the recording
    for j in range(i*lidar_clouds_per_seq, (i+1)*lidar_clouds_per_seq):
      # We get the LiDAR cloud and project it as an image
      lidar_cloud = valid_data[j][0]
      lidar_proj = compute_lidar_projection(lidar_cloud, LIDAR_MAX_RANGE, davis_intrinsics,
                                            davis_dist, t_davis_lidar)

      # We get the D_bf depth image
      bf_depth_image_raw = valid_data[j][1]
      bf_depth_image = compute_depth_image(bf_depth_image_raw, LIDAR_MAX_RANGE)

      # And the D_af depth image
      af_depth_image_raw = valid_data[j][2]
      af_depth_image = compute_depth_image(af_depth_image_raw, LIDAR_MAX_RANGE)

      # And we finish with the events, for which we compute the corresponding event volume
      evts = valid_data[j][3]
      event_volume = compute_event_volume(evts, EVTS_BINS)

      # Finally, we add the projected LiDAR cloud, the event volume, and the D_bf and D_af depth
      # images to the sequence array (the "None" is because we have no RGB data)
      sequence.append([lidar_proj, None, event_volume, bf_depth_image, af_depth_image])

    # Once all the LiDAR point clouds, events and depth images have been added to the sequence, we
    # save it, either as a compressed .pt.zip file, or as a regular .pt file
    filename = f"{prefix}_seq{i:04}.pt"
    if args.zipped:
      buffer = BytesIO()
      torch.save(sequence, buffer)
      with zipfile.ZipFile(f"{args.path_processed}/{filename}.zip", "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(filename, buffer.getvalue())
    else:
      torch.save(sequence, f"{args.path_processed}/{filename}")


def main():
  """Main function"""

  # We start by reading the args given by the user
  global args
  args = parse_args()

  # We begin by verifying that the paths given by the user are valid
  if not path.isdir(args.path_raw):
    raise FileNotFoundError("The path to the dataset should be a folder, containing .hdf5 "
                            "recordings")
  if not path.isdir(args.path_processed):
    mkdir(args.path_processed)

  # We list all the recordings in the folder
  recordings_data_paths = sorted(glob(f"{args.path_raw}/*_data.hdf5"))
  if not recordings_data_paths:
    raise ValueError("The provided folder does not contain any data file!")
  recordings_gt_paths = sorted(glob(f"{args.path_raw}/*_gt.hdf5"))
  if not recordings_gt_paths:
    raise ValueError("The provided folder does not contain any ground truth file!")
  calib_paths = sorted(glob(f"{args.path_raw}/*_calib.zip"))
  if not calib_paths:
    raise ValueError("The provided folder does not contain any calibration file!")

  # We verify that we listed the same number of _data and _gt files
  if len(recordings_data_paths) != len(recordings_gt_paths):
    raise ValueError("The provided folder does not contain the same number of data and ground "
                     "truth files!")

  # We associate the correct calibration files for the _data and _gt files
  recordings_calib_paths = []
  for data_path in recordings_data_paths:
    for calib_path in calib_paths:
      prefix = '_'.join(calib_path.split('_')[:2])
      if data_path.startswith(prefix):
        recordings_calib_paths.append(calib_path)
        break

  # We verify that we were able to associate all _data files with a calibration file
  if len(recordings_data_paths) != len(recordings_calib_paths):
    raise FileNotFoundError("Could not find all calibration files!")

  # We fuse the data, ground truth, and calibration files in a single array (required for the
  # `process_map` function)
  recordings_paths = []
  for data_path, gt_path, calib_path in zip(recordings_data_paths, recordings_gt_paths, recordings_calib_paths):
    recordings_paths.append([data_path, gt_path, calib_path])

  # Then, we process all the recordings in parallel, using the `process_map` function from tqdm
  if args.j <= 0:
    raise ValueError("The value of the -j arg must be > 0!")
  process_map(preprocess_recording, recordings_paths, max_workers=args.j)


if __name__ == "__main__":
  main()
# python preprocess_mvsec_dataset.py test /root/autodl-tmp/dataset/MVSEC/night1/ /root/autodl-tmp/dataset/MVSEC/processed/night1/ -z -j 2