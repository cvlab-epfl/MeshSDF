#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np

import lib
from lib.workspace import *
from lib.models.decoder import *
from lib.utils import *
from lib.mesh import *

import neural_renderer as nr
import pdb

AZIMUTH = 45
ELEVATION = 30
CAMERA_DISTANCE = 2.5


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Demo optimization"
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=100,
        help="The number of latent code optimization iterations to perform.",
    )
    arg_parser.add_argument(
        "--resolution",
        dest="resolution",
        default=64,
        help="Marching cubes resolution for reconstructed surfaces.",
    )
    arg_parser.add_argument(
        "--image_resolution",
        dest="image_resolution",
        default=512,
        help="Image resolution for differentiable rendering.",
    )
    arg_parser.add_argument("--fast", default=False, action="store_true" , help="Run faster iso-surface extraction algorithm presented in main paper.")

    lib.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    lib.configure_logging(args)


    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    latent_size = specs["CodeLength"]

    decoder = DeepSDF(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, model_params_subdir, "latest.pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()
    logging.info(decoder)

    optimization_meshes_dir = os.path.join(
        args.experiment_directory, optimizations_subdir
    )

    if not os.path.isdir(optimization_meshes_dir):
        os.makedirs(optimization_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        args.experiment_directory, latent_codes_subdir
    )
    latent_filename = os.path.join(
        reconstruction_codes_dir, "latest.pth"
    )
    latent = torch.load(latent_filename)["latent_codes"]["weight"]

    latent_init = latent[1]
    latent_init.requires_grad = True
    latent_target = latent[0]

    # select view point
    azimuth = AZIMUTH
    elevation = ELEVATION
    camera_distance = CAMERA_DISTANCE
    intrinsic, extrinsic = get_projection(azimuth, elevation, camera_distance, img_w=args.image_resolution, img_h=args.image_resolution)

    # set up renderer
    K_cuda = torch.tensor(intrinsic[np.newaxis, :, :].copy()).float().cuda().unsqueeze(0)
    R_cuda = torch.tensor(extrinsic[np.newaxis, 0:3, 0:3].copy()).float().cuda().unsqueeze(0)
    t_cuda = torch.tensor(extrinsic[np.newaxis, np.newaxis, 0:3, 3].copy()).float().cuda().unsqueeze(0)
    renderer = nr.Renderer(image_size = args.image_resolution, orig_size = args.image_resolution, K=K_cuda, R=R_cuda, t=t_cuda, anti_aliasing=False)

    verts_target, faces_target, _ , _ = lib.mesh.create_mesh(decoder, latent_target, N=args.resolution, output_mesh = True)

    # visualize target stuff
    verts_dr = torch.tensor(verts_target[None, :, :].copy(), dtype=torch.float32, requires_grad = False).cuda()  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces_dr = torch.tensor(faces_target[None, :, :].copy()).cuda()
    textures_dr = 0.7*torch.ones(faces_dr.shape[1], 1, 1, 1, 3, dtype=torch.float32).cuda()
    textures_dr = textures_dr.unsqueeze(0)
    image_filename = os.path.join(optimization_meshes_dir, "target.png")
    if not os.path.exists(os.path.dirname(image_filename)):
        os.makedirs(os.path.dirname(image_filename))
    tgt_images_out, tgt_depth_out, tgt_silhouette_out = renderer(verts_dr, faces_dr, textures_dr)
    store_image(image_filename, tgt_images_out, tgt_silhouette_out)

    # initialize and visualize initialization
    verts, faces, samples, next_indices = lib.mesh.create_mesh(decoder, latent_init, N=args.resolution, output_mesh = True)
    verts_dr = torch.tensor(verts[None, :, :].copy(), dtype=torch.float32, requires_grad = False).cuda()
    faces_dr = torch.tensor(faces[None, :, :].copy()).cuda()
    textures_dr = 0.7*torch.ones(faces_dr.shape[1], 1, 1, 1, 3, dtype=torch.float32).cuda()
    textures_dr = textures_dr.unsqueeze(0)
    image_filename = os.path.join(optimization_meshes_dir, "initialization.png")
    images_out, _, alpha_out = renderer(verts_dr, faces_dr, textures_dr)
    store_image(image_filename, images_out, alpha_out)

    lr= 5e-2
    regl2 = 1000
    decreased_by = 1.5
    adjust_lr_every = 500
    optimizer = torch.optim.Adam([latent_init], lr=lr)

    logging.info("Starting optimization:")
    decoder.eval()
    best_loss = None
    sigma = None
    images = []

    for e in range(args.iterations):

        optimizer.zero_grad()

        # first extract iso-surface
        if args.fast:
            verts, faces, samples, next_indices = lib.mesh.create_mesh_optim_fast(samples, next_indices, decoder, latent_init, N=args.resolution)
        else:
            verts, faces, samples, next_indices = lib.mesh.create_mesh(decoder, latent_init, N=args.resolution, output_mesh = True)

        # now assemble loss function
        xyz_upstream = torch.tensor(verts.astype(float), requires_grad = True, dtype=torch.float32, device=torch.device('cuda:0'))
        faces_upstream = torch.tensor(faces.astype(float), requires_grad = False, dtype=torch.float32, device=torch.device('cuda:0'))

        """
        Differentiable Rendering back-propagating to mesh vertices
        """

        textures_dr = 0.7*torch.ones(faces_upstream.shape[0], 1, 1, 1, 3, dtype=torch.float32).cuda()
        images_out, depth_out, silhouette_out = renderer(xyz_upstream.unsqueeze(0), faces_upstream.unsqueeze(0), textures_dr.unsqueeze(0))

        loss = torch.mean((silhouette_out-tgt_silhouette_out)**2)
        logging.info("Loss at iter {}:".format(e) + ": {}".format(loss.detach().cpu().numpy()))

        # now store upstream gradients
        loss.backward()
        dL_dx_i = xyz_upstream.grad

        # use vertices to compute full backward pass
        optimizer.zero_grad()
        xyz = torch.tensor(verts.astype(float), requires_grad = True,dtype=torch.float32, device=torch.device('cuda:0'))
        latent_inputs = latent_init.expand(xyz.shape[0], -1)

        #first compute normals
        pred_sdf = decoder(latent_inputs, xyz)
        loss_normals = torch.sum(pred_sdf)
        loss_normals.backward(retain_graph = True)
        # normalization to take into account for the fact sdf is not perfect...
        normals = xyz.grad/torch.norm(xyz.grad, 2, 1).unsqueeze(-1)
        # now assemble inflow derivative
        optimizer.zero_grad()
        dL_ds_i = -torch.matmul(dL_dx_i.unsqueeze(1), normals.unsqueeze(-1)).squeeze(-1)
        # refer to Equation (4) in the main paper
        loss_backward = torch.sum(dL_ds_i * pred_sdf)
        loss_backward.backward()
        # and update params
        optimizer.step()

        # to visualize gradients first interpolate them on face centroids
        verts_dr = torch.tensor(verts[None, :, :].copy(), dtype=torch.float32, requires_grad = False).cuda()  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
        faces_dr = torch.tensor(faces[None, :, :].copy()).cuda()
        field_faces = interpolate_on_faces(dL_ds_i, faces_dr).squeeze(1)
        # now pick a meaningful normalization, here 30% of initial grad magnitude
        if sigma is None:
            sigma = 0.3*torch.max(torch.abs(field_faces)).cpu().numpy()
        field_min = -sigma
        field_max = sigma
        field_faces = torch.clamp((field_faces-field_min)/(field_max-field_min),0,1)
        textures_dr = torch.ones(faces_dr.shape[1], 1, 1, 1, 3, dtype=torch.float32).cuda()
        # hand crafted color map
        textures_dr[:,0,0,0,0] = field_faces
        textures_dr[:,0,0,0,1] = 1.0-field_faces
        textures_dr[:,0,0,0,2] = 0.7
        textures_dr = textures_dr.unsqueeze(0)
        images_out, depth_out, alpha_out = renderer(verts_dr, faces_dr, textures_dr)
        images.append(process_image(images_out, alpha_out))

    logging.info("Optimization completed, storing GIF...")
    gif_filename = os.path.join(optimization_meshes_dir, "movie.gif")
    imageio.mimsave(gif_filename, images)
    logging.info("Done.")
