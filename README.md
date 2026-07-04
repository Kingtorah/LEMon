# Fusing LiDAR and Event Data to Estimate Dense Depth Maps: the ALED (SCIA 2023) and DELTA (CVPRW 2025) Models

![Example results of DELTA](https://vbrebion.github.io/resources/publication_teasers_github/delta.png)

This repository holds the code associated with the "Learning to Estimate Two Dense Depths from LiDAR and Event Data" article (SCIA 2023) and the "DELTA: Dense Depth from Events and LiDAR using Transformer's Attention" article (CVPRW 2025). If you use this code as part of your work, please cite:

```BibTeX
@inproceedings{Brebion2023LearningTE,
  title={Learning to Estimate Two Dense Depths from {LiDAR} and Event Data},
  author={Vincent Brebion and Julien Moreau and Franck Davoine},
  booktitle={Image Analysis - 22nd Scandinavian Conference, {SCIA} 2023, Sirkka, Finland, April 18-21, 2023, Proceedings, Part {II}},
  series={Lecture Notes in Computer Science},
  volume={13886},
  publisher={Springer},
  pages={517-533},
  year={2023}
}
```

```BibTeX
@article{Brebion2025DELTADD,
  title={{DELTA}: Dense Depth from Events and {LiDAR} using Transformer's Attention},
  author={Vincent Brebion and Julien Moreau and Franck Davoine},
  journal={Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops},
  pages={4898-4907},
  year={2025}
}
```

## Overview

In these works, we propose novel methods for fusing event and LiDAR data to estimate dense depth maps.\
With ALED, we solve the problem by using a novel convolutional architecture, able to estimate dense and accurate depth maps with high efficiency.\
With DELTA, we introduce a novel convolutional and attention-based architecture, establishing a new state of the art by being able to reduce the errors up to four times for close objects compared to ALED.\
We also introduce a novel synthetic dataset, [SLED](https://github.com/heudiasyc/SLED), allowing for a better training and evaluation of such models.

## Installation

**Note:** we recommend using `micromamba` (<https://github.com/mamba-org/mamba>) as a lighter and much faster alternative to `conda`/`miniconda`.\
However, you can safely replace `micromamba` by `conda` in the following commands if you prefer!

To install the dependencies, create a micromamba environment as follows:

```bash
micromamba create --name aled_delta
micromamba activate aled_delta
micromamba install h5py hdf5plugin matplotlib opencv pandas pytorch pyyaml tensorboard torchvision tqdm -c conda-forge
pip install fvcore standard-imghdr
```

**Note:** PyTorch version 2.6 has been used while writing/testing this code. Older/newer versions should also be compatible, but try to stick to this version if possible!

Once the environment is created, you can then clone this repository:

```bash
git clone https://github.com/heudiasyc/DELTA.git
```

## Preprocessing the datasets

To train or to test our models, you first need to preprocess the dataset(s) that you want to use. By doing so, the data formatting, normalization, LiDAR projection, ... steps are applied on the whole dataset once, rather than each time it is loaded, greatly accelerating the training and testing. The only downside is the increase in disk space usage, as each recording is converted into multiple small sequences.

To preprocess the SLED or MVSEC dataset, use the following command:

```bash
micromamba activate aled_delta
python3 dataset/preprocess/[MODEL]/preprocess_[dataset]_dataset.py [set] [path_raw] [path_processed] [-z] [-j J]
```

where:

- `[MODEL]` is the model for which the dataset is going to be used (should be `ALED` or `DELTA`);
- `[dataset]` is the name of the dataset to preprocess (should be `sled` or `mvsec`);
- `[set]` is the set that is going to be preprocessed (should be `train`, `val`, or `test`);
- `[path_raw]` is the path to the folder containing the raw dataset files **only** for the set to use;
- `[path_processed]` is the path to the output folder which will contain the preprocessed dataset files **only** for the set to use;
- `-z` should be specified if the preprocessed files should be zipped to gain disk space (this is highly recommended for SLED and MVSEC);
- `-j J` is the number of processes spawned in parallel to preprocess the dataset (avoid using high values, as this can lead to disk and memory issues; `2` is often a good choice, but do not use this argument if you are unsure (this will simply disable parallel preprocessing of the dataset)).

Preprocessing the M3ED dataset requires further dependencies, which can be installed as follows:

```bash
micromamba create --name preprocess_m3ed
micromamba activate preprocess_m3ed
micromamba install h5py "numpy<2" pandas "pip<23.2" python=3.12 pytorch tqdm -c conda-forge
pip install ouster-sdk==0.11.1
```

Then, use the following command to preprocess it:

```bash
micromamba activate preprocess_m3ed
python3 dataset/preprocess/[MODEL]/preprocess_m3ed_dataset.py [set] [path_raw] [path_processed] [-j J]
```

where the arguments are the same as described above (just avoid using `-z` with M3ED, as it can take a very long time to zip/unzip files given the size of the dataset!).

## Testing

If you only want to test ALED and/or DELTA, pretrained sets of weights (the ones used in the article) are available using [this link](https://github.com/heudiasyc/DELTA/releases/tag/v1.0).

If you rather wish to test the network after training it by yourself, see the [Training](#training) section first.

In both cases, use the following commands to test the network:

```bash
micromamba activate aled_delta
python3 test.py configs/[MODEL]/test_[dataset].json [path/to/checkpoint.pth]
python test_ledepth.py configs/DELTA/test_sled.json /root/autodl-tmp/weight/DELTA_SL.pth
python test_ledepth.py configs/LEDepth/test_sled.json /root/autodl-tmp/project/out/saves/20251216_190137/20251216_190137_080.pth

```

By default, the testing code will try to use the first GPU available. If required, you can run the testing on the GPU of your choice (for instance here, GPU 5):

```bash
CUDA_VISIBLE_DEVICES=5 python3 test.py configs/[MODEL]/test_[dataset].json [path/to/checkpoint.pth]
```


Results are saved in the `out/results/` folder, as two .txt files:

- one ending with `_per_seq.txt`, giving the detailed results for each sequence of the dataset;
- and one ending with `_global.txt`, giving a summary of the results.

## Training

If you wish to train ALED and/or DELTA, use the following commands:

```bash
micromamba activate aled_delta
OMP_NUM_THREADS=8 MASTER_ADDR="localhost" torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py configs/[MODEL]/train_[dataset].json [--cp path/to/checkpoint.pth]
OMP_NUM_THREADS=8 MASTER_ADDR="localhost" torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py configs/DELTA/train_sled.json
```

where:

- `OMP_NUM_THREADS=12` means that we should launch with 12 threads (since we use 12 workers, as set up in the config files);
- `MASTER_ADDR="localhost"` means that the training should be running on the server it is launched from;
- `torchrun` is the utility used for being able to train on more than 1 GPU; more details on multi-GPU training in PyTorch [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html), and on torchrun [here](https://pytorch.org/docs/stable/elastic/run.html);
- `--standalone --nnodes=1 --nproc-per-node=2` are torchrun parameters to indicate that we use a single server with 2 GPUs (as specified [in the documentation](https://pytorch.org/docs/stable/elastic/run.html#single-node-multi-worker));
- `train.py` is the Python script to use for training;
- `configs/[MODEL]/train_[dataset].json` is the only mandatory argument for the training script, giving the path to the config file to use;
- `[--cp path/to/checkpoint.pth]` is an optional argument, allowing for restarting the training from a pretrained checkpoint.

If you wish to follow the progress of the training, all data (losses, validation error, images, ...) is saved in Tensorboard. To display it, open a second terminal and use the following command:

```bash
micromamba activate aled_delta
tensorboard --logdir out/runs/ --samples_per_plugin "images=4000"
```

You can then open the web browser of your choice, and use the `http://localhost:6006/` address to access the results. If needed, you can also adapt the number of images if you want to adjust the memory usage of Tensorboard.

## Code details

The code was developed to be as simple and as clean as possible. For a better comprehension especially, each network was split into submodules, matching the way the model architectures are described in their reference article.

Each file and each function was properly documented, so do not hesitate to take a look at them!
