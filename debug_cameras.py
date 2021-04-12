#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time
import pdb
import imageio
import numpy as np

import lib
import lib.workspace as ws
from lib.utils import *

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    FoVPerspectiveCameras,
    OpenGLOrthographicCameras,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
    TexturesAtlas
)
from pytorch3d.ops import sample_points_from_meshes
device=torch.device('cuda:0')

def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def main_function(experiment_directory, continue_from):

    logging.debug("running " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + ' '.join([str(elem) for elem in specs["Description"]]))

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]
    test_split_file = specs["TestSplit"]

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    with open(test_split_file, "r") as f:
        test_split = json.load(f)
    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 16)
    logging.info("loading data with {} threads".format(num_data_loader_threads))
    sdf_dataset_test = lib.data.RGBA2SDF(
        data_source, test_split, num_samp_per_scene, is_train=False, num_views = specs["NumberOfViews"]
    )
    sdf_loader_test = data_utils.DataLoader(
        sdf_dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    out_dir = "./experiments/scratch"

    for sdf_data, image, intrinsic, extrinsic, name in sdf_loader_test:

        with torch.no_grad():
            # Process the input data
            xyz = sdf_data[:, :, 0:3].cuda()
            sdf_gt = sdf_data[:, :, 3].cuda()

            # store debug images
            out_name = os.path.join(out_dir, name[0].split("/")[-1])

            intrinsic_np = intrinsic[0].numpy()
            extrinsic_np = extrinsic[0].numpy()
            # select only points inside shape
            xyz_in = xyz[sdf_gt<=0].cpu().numpy()
            # pre-processing, lift coordinates to homogeneous to write roto-translation compactly
            xyz_homo = np.concatenate((xyz_in, np.ones((xyz_in.shape[0],1),dtype=np.float32)),axis=-1)
            # 1) map from world to camera coordinates
            xyz_camera = np.dot(xyz_homo, extrinsic_np.T)
            # 2) map from camera to image coordinates
            xyz_image = np.dot(xyz_camera, intrinsic_np.T)
            xy = xyz_image[:,:2] / np.expand_dims(xyz_image[:,2], axis=1)

            # now take input image and see where points projects
            image_np = 255*image.detach().cpu().numpy()[0].transpose((1, 2, 0))[...,0:3]

            for j in range(xy.shape[0]):
                y = int(xy[j, 1])
                x = int(xy[j, 0])
                try:
                    image_np[y,x] = [255,0,0]
                except:
                    continue

            image_out_filename = out_name + "_samples.png"
            imageio.imwrite(image_out_filename, image_np.astype(np.uint8))


            # load mesh and map to ShapeNet convention
            mesh_filename =  "data/" + name[0].split("/")[0] + "/meshes/" + name[0].split("/")[-1] + ".obj"

            mesh = trimesh.load(mesh_filename)
            verts_DISN = torch.tensor(mesh.vertices).float()
            # now we wanna map them to shapenet convention

            vertices = verts_DISN.cuda().unsqueeze(0)
            faces = torch.tensor(mesh.faces).float().cuda().unsqueeze(0)
            meshes = Meshes(vertices, faces)
            verts_shape = meshes.verts_packed().shape
            sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=False)
            meshes.textures = TexturesVertex(verts_features=sphere_verts_rgb)

            pts_np = sample_points_from_meshes(meshes).squeeze(0).detach().cpu().numpy()
            # pre-processing, lift coordinates to homogeneous to write roto-translation compactly
            xyz_homo = np.concatenate((pts_np, np.ones((pts_np.shape[0],1),dtype=np.float32)),axis=-1)
            # 1) map from world to camera coordinates
            xyz_camera = np.dot(xyz_homo, extrinsic_np.T)
            # 2) map from camera to image coordinates
            xyz_image = np.dot(xyz_camera, intrinsic_np.T)
            xy = xyz_image[:,:2] / np.expand_dims(xyz_image[:,2], axis=1)

            # now take input image and see where points projects
            image_np = 255*image.detach().cpu().numpy()[0].transpose((1, 2, 0))[...,0:3]

            for j in range(xy.shape[0]):
                y = int(xy[j, 1])
                x = int(xy[j, 0])
                try:
                    image_np[y,x] = [0,0,255]
                except:
                    continue

            image_out_filename = out_name + "_mesh.png"
            imageio.imwrite(image_out_filename, image_np.astype(np.uint8))


            IMG_SIZE = image.shape[-1]
            sigma = 1e-5

            K_cuda = torch.tensor(intrinsic[:, 0:3, 0:3]).float().cuda()
            # X_cam = X_world * R + T and NOT X_cam = R * X_world + T
            R_cuda = torch.tensor(extrinsic[:, 0:3, 0:3]).float().cuda().permute(0,2,1)
            t_cuda = torch.tensor(extrinsic[:, 0:3, 3]).float().cuda()

            lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
            cameras = PerspectiveCameras(device=device, focal_length=-K_cuda[:,0,0], principal_point=((K_cuda[:,0,2], K_cuda[:,1,2]),), image_size=((IMG_SIZE, IMG_SIZE),), R=R_cuda, T=t_cuda)

            raster_settings_soft = RasterizationSettings(
                image_size=IMG_SIZE,
                blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
                faces_per_pixel=50,
            )

            # silhouette renderer
            silhouette_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings_soft
                ),
                shader=SoftSilhouetteShader()
            )

            buffer = silhouette_renderer(meshes_world=meshes, cameras=cameras, lights=lights)
            silhouette = buffer[..., 3]

            image_out_export = 255*silhouette.detach().cpu().numpy()[0]
            image_out_filename = image_out_filename = out_name + "_silo.png"
            imageio.imwrite(image_out_filename, image_out_export.astype(np.uint8))

            print("Done w ", name[0])




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
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )

    lib.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    lib.configure_logging(args)

    main_function(args.experiment_directory, args.continue_from)
