#!/usr/bin/env python3

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import pdb
import imageio
import matplotlib
import time
from scipy.spatial.transform import Rotation as R
import random
import torchvision
import imageio
from PIL import Image
import h5py
import json
import time

from lib.utils import *


def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, instance_filename)
                ):
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def unpack_sdf_samples(filename, subsample=None):

    npz = np.load(filename)
    if subsample is None:
        return npz

    pos_tensor = remove_nans(torch.from_numpy(npz["pos"].astype(float)))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"].astype(float)))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half).cpu() * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half).cpu() * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0).float()

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample
    ):
        self.subsample = subsample
        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)


    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        return unpack_sdf_samples(filename, self.subsample), idx, self.npyfiles[idx]
