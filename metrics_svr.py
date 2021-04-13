#!/usr/bin/env python3

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import json
import time
import pdb

import imageio
import numpy as np

import lib
import lib.workspace as ws
from lib.utils import *


def main_function(experiment_directory, continue_from):

    device=torch.device('cuda:0')
    specs = ws.load_experiment_specifications(experiment_directory)

    print("Recapping metrics from experiment description: \n" + ' '.join([str(elem) for elem in specs["Description"]]))

    data_source = specs["DataSource"]
    test_split_file = specs["TestSplit"]

    num_samp_per_scene = specs["SamplesPerScene"]
    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    sdf_dataset_test = lib.data.RGBA2SDF(
        data_source, test_split, num_samp_per_scene, is_train=False, num_views = specs["NumberOfViews"]
    )
    sdf_loader_test = data_utils.DataLoader(
        sdf_dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )
    num_scenes = len(sdf_loader_test)
    print("There are {} scenes".format(num_scenes))

    optimization_meshes_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(continue_from)
    )

    log_silhouettes = []
    log_chds = []
    log_ncs = []
    log_latents = []

    for sdf_data, image, intrinsic, extrinsic, name in sdf_loader_test:
        out_name = os.path.join(optimization_meshes_dir,name[0].split("/")[-1])
        # load all logs
        log_filename = os.path.join(out_name,  "log_silhouette.npy")
        log_silhouette = np.load(log_filename)
        log_filename = os.path.join(out_name, "log_chd.npy")
        log_chd = np.load(log_filename)
        log_filename = os.path.join(out_name, "log_nc.npy")
        log_nc = np.load(log_filename)
        log_filename = os.path.join(out_name, "log_latent.npy")
        log_latent = np.load(log_filename)
        # accumulate logs
        log_silhouettes.append(log_silhouette)
        log_chds.append(log_chd)
        log_ncs.append(log_nc)
        log_latents.append(log_latent)

    print('Stats:')
    log_silhouette = np.mean(np.stack(log_silhouettes), axis=0)
    log_chd = np.mean(np.stack(log_chds), axis=0)
    log_nc = np.mean(np.stack(log_ncs), axis=0)
    log_latent = np.mean(np.stack(log_latents), axis=0)

    print('Raw CHD {:.2f}'.format( 1000*log_chd[0]))
    print('Refined CHD {:.2f}'.format( 1000*log_chd[-1]))
    print('Improvement in CHD {:.2f} %'.format( 100*(log_chd[0] - log_chd[-1])/log_chd[0] ))

    print('Raw NC {:.2f} %'.format( 100*(log_nc[0])))
    print('Refined NC {:.2f} %'.format( 100*(log_nc[-1])))
    print('Improvement in NC {:.2f} %'.format( 100*(log_nc[-1] - log_nc[0])/log_nc[0] ))



if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        default="latest",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )

    args = arg_parser.parse_args()
    main_function(args.experiment_directory, args.continue_from)
