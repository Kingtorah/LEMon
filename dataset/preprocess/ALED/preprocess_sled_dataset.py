#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script can be called to preprocess the SLED dataset for the ALED model, transforming the raw
.npz recordings into PyTorch .pt files, with all the preprocessing steps (data formatting,
normalization, ...) already applied.
Preprocessing the dataset has its pros and cons:
- it normalizes the datasets, so that they can all be loaded using a single dataloader;
- it reduces greatly the computational power necessary to load the dataset during training /
  validation / testing;
- it greatly increases disk space usage, as the dataset is converted into multiple small sequences;
- if the preprocessing steps are modified, the whole dataset has to be preprocessed again.
"""

from argparse import ArgumentParser, Namespace
import csv
from io import BytesIO
from os import mkdir, path
import zipfile

import numpy as np
from numpy import ndarray
import pandas as pd
import torch
from torch import Tensor
from tqdm.contrib.concurrent import process_map


LIDAR_MAX_RANGE = 200
EVTS_BINS = 5
RGB_DVS_FOV = 90

# We define args as a global variable, so that it can be seen by all parallel processes
args = Namespace()


def parse_args():
  """Args parser"""
  parser = ArgumentParser()
  parser.add_argument("set", help="Should be 'train', 'val', or 'test'")
  parser.add_argument("path_raw", help="Path to the folder containing the raw sequences to treat")
  parser.add_argument("path_processed", help="Path to the folder where the dataset will be stored "
    "after preprocessing")
  parser.add_argument("-z", "--zipped", action="store_true", help="If this flag is set, the "
    "preprocessed sequences will be compressed as .pt.zip files (otherwise, they are saved as "
    "regular .pt files)")
  parser.add_argument("-j", type=int, default=1, help="Number of parallel processes spawned to "
    "preprocess the dataset. If not specified, a single process is spawned")
  return parser.parse_args()


def compute_lidar_projection(lidar_cloud: ndarray, lidar_max_range: int, camera_fov: int) -> Tensor:
  """
  Creates a projection of a point cloud in a 1-channel Tensor.
  Each pixel contains its depth value, normalized between 0 and 1 (pixels with no LiDAR point are
  given a value of 0).
  """

  # We create a false camera, of resolution 1280x720, aligned with the LiDAR sensor
  # R_c_l is the rotation matrix from LiDAR to camera, to correct the axes
  f = 1280/(2*np.tan(camera_fov*np.pi/360))
  cx = 1280/2
  cy = 720/2
  K = np.array([[f, 0, cx],
                [0, f, cy],
                [0, 0, 1 ]])
  R_c_l = np.array([[0, 1, 0],
                    [0, 0, -1],
                    [1, 0, 0]])

  # We then filter the point cloud, to only retain points in front of the camera
  lidar_cloud_filt = lidar_cloud[lidar_cloud[:, 0] > 0]
  pcl_pts_filt = lidar_cloud_filt[:, :3]

  # We project them to the camera frame
  pcl_camera_frame = (R_c_l @ pcl_pts_filt.T).T
  depths = pcl_camera_frame[:, 2].copy()
  pcl_camera_frame[:, 0] /= depths
  pcl_camera_frame[:, 1] /= depths
  pcl_camera_frame[:, 2] /= depths

  # We project them in the image
  pcl_camera = (K @ pcl_camera_frame.T).T

  # We create the projection, and add each projected LiDAR point to it
  # The projection is composed of 1 channel, containing the normalized depth of each point
  lidar_proj = torch.zeros(1, 720, 1280)
  for i, pt in enumerate(pcl_camera[:, :2]):
    if pt[0] >= 0 and pt[0] < 1280 and pt[1] >= 0 and pt[1] < 720:
      lidar_proj[0, int(pt[1]), int(pt[0])] = min(depths[i]/lidar_max_range, 1.0)

  # We return the projection
  return lidar_proj


def compute_event_volume(events: ndarray, bins: int) -> Tensor:
  """
  From a numpy array of events, computes an event volume, as described in the "Learning to Detect
  Objects with a 1 Megapixel Event Camera" article by Perot et al.
  This implementation is optimized for fast computation (which is still a bit slow :c), thanks to
  https://stackoverflow.com/a/55739936
  """

  # We create an empty event volume
  event_volume = np.zeros((2*bins, 720, 1280), np.float32)

  # We compute the t_star value for each event
  t_star = (bins-1)*(events["t"]-events[0]["t"])/(events[-1]["t"]-events[0]["t"])

  # We create an index of unique (x, y, pol) events
  idx, u_evts = pd.factorize(events[["x", "y", "pol"]])

  # Then, for each bin...
  for i in range(bins):
    # We compute the sum of the max(0, 1-abs(bin-t_star)) for each pixel
    sums = np.bincount(idx, np.fmax(0, 1-abs(i-t_star)))

    # We set these values inside the event volume
    event_volume[i+bins*u_evts["pol"], u_evts["y"], u_evts["x"]] = sums

  # We finally return the event volume, in the PyTorch format
  return torch.from_numpy(event_volume)


def compute_depth_image(depth_image_raw: ndarray, lidar_max_range: int) -> Tensor:
  """
  From a raw CARLA depth image, computes its equivalent Tensor-based representation.
  Details on how the conversion works can be found here:
  https://carla.readthedocs.io/en/0.9.13/ref_sensors/#depth-camera
  """

  # We convert the raw depth image to a float32 matrix of depth values in meters
  depth_image = depth_image_raw.astype(np.float32)
  depth_image = ((depth_image[:, :, 2] + depth_image[:, :, 1]*256 + depth_image[:, :, 0]*256*256) /
                 (256*256*256 - 1))
  depth_image *= 1000

  # We normalize these values based on the max range of the LiDAR
  # Note that the depth image contains values > than 1.0, which should probably be filtered out
  # during training
  depth_image /= lidar_max_range

  # Finally, we transform the numpy matrix to a PyTorch Tensor
  depth_image = torch.from_numpy(depth_image)
  depth_image = depth_image.unsqueeze(0)

  # And we return it
  return depth_image


def preprocess_recording(recording_path: str) -> None:
  """
  Recording preprocessing function, which can be called in parallel on all the recordings.
  """

  # We open and read data from the file
  recording = np.load(args.path_raw+"/"+recording_path, allow_pickle=True)
  lidar_clouds_with_ts = recording["lidar_clouds"]
  events_with_ts = recording["events"]
  depth_images_with_ts = recording["depth_images"]

  # The number of LiDAR images per sequence is determined based on the set in use:
  # - 3 for the train set
  # - 20 for the validation set
  # - all for the test set (we keep the full recording, without any splitting)
  if args.set == "train":
    lidar_clouds_per_seq = 3
  elif args.set == "val":
    lidar_clouds_per_seq = 20
  else:
    lidar_clouds_per_seq = len(lidar_clouds_with_ts) - 1

  # Each recording allows us to generate N/L sequences ((N/L)-1 if N%L==0), where N is the total
  # number of point clouds in the recording, and L is the total number of successive point clouds in
  # a sequence
  if len(lidar_clouds_with_ts) % lidar_clouds_per_seq == 0:
    nb_seq = len(lidar_clouds_with_ts) // lidar_clouds_per_seq - 1
  else:
    nb_seq = len(lidar_clouds_with_ts) // lidar_clouds_per_seq

  # We generate each of the nb_seq sequences
  for i in range(nb_seq):
    sequence = []

    # Each sequence contains L successive LiDAR clouds from the recording
    for j in range(i*lidar_clouds_per_seq, (i+1)*lidar_clouds_per_seq):
      # We extract the LiDAR cloud, its timestamp, and project it as an image
      lidar_cloud, start_ts = lidar_clouds_with_ts[j]
      lidar_proj = compute_lidar_projection(lidar_cloud, LIDAR_MAX_RANGE, RGB_DVS_FOV)

      # Since the LiDAR in CARLA still doesn't see some objects (even though it is supposed to be
      # fixed, see https://github.com/carla-simulator/carla/issues/5732), we replace points with a
      # distance computed from the LiDAR with the distance from the depth map directly.
      # If it is fixed one day in a new release / new version of SLED, remove this paragraph of code
      depth_image_raw = depth_images_with_ts[depth_images_with_ts[:, 1] >= start_ts][0, 0]
      depth_image = compute_depth_image(depth_image_raw, LIDAR_MAX_RANGE)
      mask = torch.bitwise_and(lidar_proj[0, :, :] != 0, depth_image[0, :, :] < 1.0)
      lidar_proj[0, :, :][mask] = depth_image[0, :, :][mask]
      lidar_proj[0, :, :][~mask] = 0.

      # To know which events and depth images should be extracted, we set the end timestamp as the
      # timestamp of the next LiDAR scan
      end_ts = lidar_clouds_with_ts[j+1, 1]

      # We extract the event arrays based on this timestamp range
      events_ts_mask = np.bitwise_and(events_with_ts[:, 1] > start_ts,
                                      events_with_ts[:, 1] <= end_ts)
      events = events_with_ts[events_ts_mask][:, 0]

      # We concatenate them to have 2 event arrays per LiDAR cloud (so, 50ms of events for a 10Hz
      # LiDAR, for instance)
      nb_events = events.shape[0]
      events_concat = [np.concatenate(events[0*nb_events//2:1*nb_events//2], axis=None),
                       np.concatenate(events[1*nb_events//2:2*nb_events//2], axis=None)]

      # And for each of them, we compute the corresponding event volume
      event_volumes = []
      for event_array in events_concat:
        event_volume = compute_event_volume(event_array, EVTS_BINS)
        event_volumes.append(event_volume)

      # We do the same with the D_bf depth images
      bf_depth_images_ts_mask = np.bitwise_and(depth_images_with_ts[:, 1] >= start_ts,
                                               depth_images_with_ts[:, 1] <= end_ts)
      bf_depth_images_raw = depth_images_with_ts[bf_depth_images_ts_mask][:, 0]
      bf_depth_images_raw_restricted = [bf_depth_images_raw[0],
                                        bf_depth_images_raw[bf_depth_images_raw.shape[0]//2]]
      bf_depth_images = []
      for bf_depth_image_raw in bf_depth_images_raw_restricted:
        bf_depth_image = compute_depth_image(bf_depth_image_raw, LIDAR_MAX_RANGE)
        bf_depth_images.append(bf_depth_image)

      # And the D_af depth images
      af_depth_images_ts_mask = np.bitwise_and(depth_images_with_ts[:, 1] >= start_ts,
                                               depth_images_with_ts[:, 1] <= end_ts)
      af_depth_images_raw = depth_images_with_ts[af_depth_images_ts_mask][:, 0]
      af_depth_images_raw_restricted = [af_depth_images_raw[af_depth_images_raw.shape[0]//2],
                                        af_depth_images_raw[-1]]
      af_depth_images = []
      for af_depth_image_raw in af_depth_images_raw_restricted:
        af_depth_image = compute_depth_image(af_depth_image_raw, LIDAR_MAX_RANGE)
        af_depth_images.append(af_depth_image)

      # Finally, we add the projected LiDAR cloud, the event volumes, the RGB images, and the D_bf
      # and D_af depth images to the sequence array
      sequence.append([lidar_proj, None, event_volumes[0], bf_depth_images[0], af_depth_images[0]])
      sequence.append([None,       None, event_volumes[1], bf_depth_images[1], af_depth_images[1]])

    # Once all the LiDAR point clouds, events and depth images have been added to the sequence, we
    # save it, either as a compressed .pt.zip file, or as a regular .pt file
    filename = f"{recording_path[:-4]}_seq{i:04}.pt"
    if args.zipped:
      buffer = BytesIO()
      torch.save(sequence, buffer)
      with zipfile.ZipFile(f"{args.path_processed}/{filename}.zip", "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(filename, buffer.getvalue())
    else:
      torch.save(sequence, f"{args.path_processed}/{filename}")

  # Once the recording has been fully explored, we don't forget to close it
  recording.close()


def main():
  """Main function"""

  # We start by reading the args given by the user
  global args
  args = parse_args()

  # We begin by verifying that the paths given by the user are valid
  if not path.isdir(args.path_raw):
    raise FileNotFoundError("The path to the dataset should be a folder, containing .npz "
      "recordings and a metadata.csv file")
  if not path.isdir(args.path_processed):
    mkdir(args.path_processed)

  # We also check that the given set is valid
  if args.set not in {"train", "val", "test"}:
    raise ValueError(f"Invalid value '{args.set}' for the 'set' argument")

  # Based on the metadata.csv file, we list all the recordings in the folder
  recordings_paths = []
  with open(args.path_raw+"/metadata.csv", encoding="utf-8", newline='') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    for row in csv_reader:
      recordings_paths.append(row[0])
  if not recordings_paths:
    raise ValueError("The provided metadata.csv file is empty!")

  # Then, we process all the recordings in parallel, using the `process_map` function from tqdm
  if args.j <= 0:
    raise ValueError("The value of the -j arg must be > 0!")
  process_map(preprocess_recording, recordings_paths, max_workers=args.j)


if __name__ == "__main__":
  main()
