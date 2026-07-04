#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script can be called to preprocess the M3ED dataset for the DELTA model, transforming the raw
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
from ouster.sdk.client import LidarPacket, Packets, Scans, SensorInfo, XYZLut
import pandas as pd
import torch
from torch import Tensor
from tqdm.contrib.concurrent import process_map


LIDAR_MAX_RANGE = 120
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


def compute_lidar_projection(lidar_cloud: ndarray, lidar_max_range: int,
                             prophesee_intrinsics: ndarray, prophesee_dist: ndarray,
                             t_prophesee_lidar: ndarray) -> Tensor:
  """
  Creates a projection of the point cloud in a 1-channel Tensor.
  Each pixel contains its depth value, normalized between 0 and 1 (pixels with no LiDAR point are
  given a value of 0).
  """

  # We use the calibration parameters to initialize the camera matrix
  K = np.array([[prophesee_intrinsics[0], 0, prophesee_intrinsics[2]],
                [0, prophesee_intrinsics[1], prophesee_intrinsics[3]],
                [0, 0, 1]])

  # We then filter the point cloud, to only retain points in front of the LiDAR
  lidar_cloud_filt = lidar_cloud[lidar_cloud[:, 0] > 0]

  # We add a "1" at the end of each row to be able to use homogeneous coordinates
  lidar_cloud_filt = np.concatenate((lidar_cloud_filt, np.ones((lidar_cloud_filt.shape[0], 1))),
                                    axis=1)

  # We tranform the coordinates to the camera's frame
  pcl_camera_frame = ((t_prophesee_lidar @ lidar_cloud_filt.T).T)[:, :3]
  pcl_camera_frame_filt = pcl_camera_frame[pcl_camera_frame[:, 2] >= 1.0]
  depths = pcl_camera_frame_filt[:, 2].copy()

  # We apply the distortions, using the classical radtan model, and we project the points in the
  # image plane
  # Note: the following code is equivalent to the one of cv2.projectPoints(), but much lighter
  pcl_camera_frame_filt[:, 0] /= depths
  pcl_camera_frame_filt[:, 1] /= depths
  pcl_camera_frame_filt[:, 2] /= depths
  pcl_camera_frame_dist = np.ones_like(pcl_camera_frame_filt)
  x = pcl_camera_frame_filt[:, 0]
  y = pcl_camera_frame_filt[:, 1]
  k1, k2, p1, p2 = prophesee_dist
  r_2 = x*x + y*y
  r_4 = r_2*r_2
  pcl_camera_frame_dist[:, 0] = x * (1 + k1*r_2 + k2*r_4) + \
                                2 * p1 * x * y + \
                                p2 * (r_2 + 2 * x * x)
  pcl_camera_frame_dist[:, 1] = y * (1 + k1*r_2 + k2*r_4) + \
                                p1 * (r_2 + 2 * y * y) + \
                                2 * p2 * x * y
  pcl_camera = (K @ pcl_camera_frame_dist.T).T

  # We create the projection, and add each projected LiDAR point to it
  lidar_proj = torch.zeros(1, 720, 1280)
  for i, pt in enumerate(pcl_camera[:, :2]):
    if pt[0] >= 0 and pt[0] < 1280 and pt[1] >= 0 and pt[1] < 720:
      lidar_proj[0, int(pt[1]), int(pt[0])] = min(depths[i]/lidar_max_range, 1.0)

  # We return the projection
  return lidar_proj


def compute_event_volume(events_x: ndarray, events_y: ndarray, events_t: ndarray, events_p: ndarray,
                         bins: int) -> Tensor:
  """
  From numpy arrays of events (x/y/t/p), computes an event volume, as described in the "Unsupervised
  Event-based Learning of Optical Flow, Depth, and Egomotion" article by Zhu et al.
  This implementation is optimized for fast computation (which is still a bit slow :c), thanks to
  https://stackoverflow.com/a/55739936
  """

  # We create an empty event volume
  event_volume = np.zeros((bins, 720, 1280), np.float32)

  # We compute the t_star value for each event
  t_star = (bins-1)*(events_t-events_t[0])/(events_t[-1]-events_t[0])

  # We create an index of unique (x, y) events
  evts_x_y = list(zip(events_x, events_y))
  idx, u_evts = pd.factorize(evts_x_y)
  u_evts = np.array([np.array(u_evt) for u_evt in u_evts])

  # Then, for each bin...
  for i in range(bins):
    # We compute the sum of the pol*max(0, 1-abs(bin-t_star)) for each pixel
    # Note that polarities in M3ED are represented as 0/1, so we convert them to -1/+1
    sums = np.bincount(idx, (2*events_p-1)*np.fmax(0, 1-abs(i-t_star)))

    # We set these values inside the event volume
    event_volume[i, u_evts[:, 1], u_evts[:, 0]] = sums

  # We finally return the event volume, in the PyTorch format
  return torch.from_numpy(event_volume)


def compute_depth_image(depth_image_raw: ndarray, lidar_max_range: int) -> Tensor:
  """
  From a raw depth image, computes its equivalent Tensor-based representation
  """

  # We replace the "Inf" values by "NaN", to mimic the behaviour of the MVSEC dataset
  depth_image = depth_image_raw.copy()
  depth_image[depth_image == np.inf] = np.nan

  # We normalize the values based on the max range of the LiDAR
  # Note that the depth image contains values > than 1.0, which should probably be filtered out
  # during training
  depth_image /= lidar_max_range

  # Finally, we transform the numpy matrix to a PyTorch Tensor
  depth_image = torch.from_numpy(depth_image)

  # And we return it
  return depth_image


def compute_lidar_cloud_from_ouster_packets(lidar_packet_buf, lidar_info: SensorInfo) -> ndarray:
  """
  Converts the LiDAR packets in the native Ouster format into a LiDAR point cloud as a numpy array.
  This code is inspired from:
  https://github.com/daniilidis-group/m3ed/blob/main/build_system/lidar_depth/util.py#L243
  """
  lidar_packets = [LidarPacket(buf, lidar_info) for buf in lidar_packet_buf]
  lidar_scans = Scans(Packets(lidar_packets, lidar_info))
  lidar_scan = next(iter(lidar_scans))
  lidar_metadata = lidar_scans.metadata
  lidar_xyzlut = XYZLut(lidar_metadata)(lidar_scan)
  [lidar_x, lidar_y, lidar_z] = [c.flatten() for c in np.dsplit(lidar_xyzlut, 3)]
  lidar_cloud = np.array((lidar_x, lidar_y, lidar_z)).T
  return lidar_cloud


def preprocess_recording(paths: list[str]) -> None:
  """
  Recording preprocessing function, which can be called in parallel on all the recordings.
  """

  # We get the prefix of the file (car_urban_day_city_hall, ...)
  prefix = paths[0].split('/')[-1][:-8]

  # We open the _data and _gt files
  data_recording = h5py.File(paths[0])
  gt_recording = h5py.File(paths[1])

  # We read the event data
  prophesee_x = data_recording["/prophesee/left/x"][()]
  prophesee_y = data_recording["/prophesee/left/y"][()]
  prophesee_t = data_recording["/prophesee/left/t"][()]
  prophesee_p = data_recording["/prophesee/left/p"][()]

  # We read the LiDAR data
  lidar_packets_buf = data_recording['/ouster/data'][()]
  lidar_info = SensorInfo(data_recording['/ouster/metadata'][()])

  # We read the GT depth maps, and compute their quantity
  # Note: since we need depth maps before and after the events, we cannot use the last depth map as
  # a "before" as it would not have an "after"
  gt_depth_maps = gt_recording["/depth/prophesee/left"][()]
  nb_bf_gt_depth_maps = gt_depth_maps.shape[0] - 1

  # We read the calibration data
  prophesee_intrinsics = data_recording["/prophesee/left/calib/intrinsics"][()]
  prophesee_dist = data_recording["/prophesee/left/calib/distortion_coeffs"][()]
  t_prophesee_lidar = data_recording["/ouster/calib/T_to_prophesee_left"][()]

  # The number of LiDAR images per sequence is determined based on the set in use:
  # - 3 for the train set
  # - 20 for the validation set
  # - all for the test set (we keep the full recording, without any splitting)
  if args.set == "train":
    lidar_clouds_per_seq = 3
  elif args.set == "val":
    lidar_clouds_per_seq = 20
  else:
    lidar_clouds_per_seq = nb_bf_gt_depth_maps

  # Each recording allows us to generate N/L sequences, where N is the total of "before" depth maps,
  # and L is the total number of successive point clouds (= nb of bf depth maps) in a sequence
  nb_seq = nb_bf_gt_depth_maps // lidar_clouds_per_seq

  # We generate each of the nb_seq sequences
  for i in range(nb_seq):
    sequence = []

    # Each sequence contains L successive LiDAR clouds from the recording
    for j in range(i*lidar_clouds_per_seq, (i+1)*lidar_clouds_per_seq):
      # We get the LiDAR cloud and project it as an image
      lidar_packet_buf = lidar_packets_buf[j, :, :]
      lidar_cloud = compute_lidar_cloud_from_ouster_packets(lidar_packet_buf, lidar_info)
      lidar_proj = compute_lidar_projection(lidar_cloud, LIDAR_MAX_RANGE, prophesee_intrinsics,
                                            prophesee_dist, t_prophesee_lidar)

      # We get the D_bf depth image
      bf_depth_image_raw = gt_depth_maps[[j], :, :]
      bf_depth_image = compute_depth_image(bf_depth_image_raw, LIDAR_MAX_RANGE)

      # And the D_af depth image
      af_depth_image_raw = gt_depth_maps[[j+1], :, :]
      af_depth_image = compute_depth_image(af_depth_image_raw, LIDAR_MAX_RANGE)

      # And we finish with the events, for which we compute the corresponding event volumes
      events_start_idx = gt_recording["/ts_map_prophesee_left"][j]
      events_end_idx = gt_recording["/ts_map_prophesee_left"][j+1]
      events_mid_idx = int((events_end_idx+events_start_idx)//2)
      events_x_0 = prophesee_x[events_start_idx:events_mid_idx]
      events_y_0 = prophesee_y[events_start_idx:events_mid_idx]
      events_t_0 = prophesee_t[events_start_idx:events_mid_idx]
      events_p_0 = prophesee_p[events_start_idx:events_mid_idx]
      events_x_1 = prophesee_x[events_mid_idx:events_end_idx]
      events_y_1 = prophesee_y[events_mid_idx:events_end_idx]
      events_t_1 = prophesee_t[events_mid_idx:events_end_idx]
      events_p_1 = prophesee_p[events_mid_idx:events_end_idx]
      event_volumes = [compute_event_volume(events_x_0, events_y_0, events_t_0, events_p_0, EVTS_BINS),
                       compute_event_volume(events_x_1, events_y_1, events_t_1, events_p_1, EVTS_BINS)]

      # Finally, we add the projected LiDAR cloud, the event volume, and the D_bf and D_af depth
      # images to the sequence array
      # Since the GT in M3ED is only available at 10Hz, we have to add "None" GT images to reach a
      # GT at 20Hz (such that events can be accumulated over 50ms, to match SLED and MVSEC)
      sequence.append([lidar_proj, None, event_volumes[0], bf_depth_image, None           ])
      sequence.append([None,       None, event_volumes[1], None,           af_depth_image])

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
    raise FileNotFoundError("The path to the dataset should be a folder, containing .h5 recordings")
  if not path.isdir(args.path_processed):
    mkdir(args.path_processed)

  # We list all the recordings in the folder
  recordings_data_paths = sorted(glob(f"{args.path_raw}/*_data.h5"))
  if not recordings_data_paths:
    raise ValueError("The provided folder does not contain any data file!")
  recordings_gt_paths = sorted(glob(f"{args.path_raw}/*_depth_gt.h5"))
  if not recordings_gt_paths:
    raise ValueError("The provided folder does not contain any ground truth file!")

  # We verify that we listed the same number of _data and _gt files
  if len(recordings_data_paths) != len(recordings_gt_paths):
    raise ValueError("The provided folder does not contain the same number of data and ground "
                     "truth files!")

  # We fuse the data and ground truth files in a single array (required for the `process_map`
  # function)
  recordings_paths = []
  for data_path, gt_path in zip(recordings_data_paths, recordings_gt_paths):
    recordings_paths.append([data_path, gt_path])

  # Then, we process all the recordings in parallel, using the `process_map` function from tqdm
  if args.j <= 0:
    raise ValueError("The value of the -j arg must be > 0!")
  process_map(preprocess_recording, recordings_paths, max_workers=args.j)


if __name__ == "__main__":
  main()
