#!/usr/bin/env python3

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
import pdb

from lib.utils import *

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function is taken from https://github.com/facebookresearch/DeepSDF
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file
    write_verts_faces_to_file(verts, faces, ply_filename_out)
    

def write_verts_faces_to_file(verts, faces, ply_filename_out):

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)


def convert_sdf_samples_to_mesh(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to vertices,faces

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function is adapted from https://github.com/facebookresearch/DeepSDF
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # return mesh
    return mesh_points, faces


def create_mesh(
    decoder, latent_vec, N=256, max_batch=32 ** 3, offset=None, scale=None, output_mesh = False, filename = None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        samples[head : min(head + max_batch, num_samples), 3] = (
            decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    if output_mesh is False:

        convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            ply_filename + ".ply",
            offset,
            scale,
        )
        return

    else:

        verts, faces = convert_sdf_samples_to_mesh(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            offset,
            scale,
        )

        # first fetch bins that are activated
        k = ((verts[:, 2] -  voxel_origin[2])/voxel_size).astype(int)
        j = ((verts[:, 1] -  voxel_origin[1])/voxel_size).astype(int)
        i = ((verts[:, 0] -  voxel_origin[0])/voxel_size).astype(int)
        # find points around
        next_samples = i*N*N + j*N + k
        next_samples_ip = np.minimum(i+1,N-1)*N*N + j*N + k
        next_samples_jp = i*N*N + np.minimum(j+1,N-1)*N + k
        next_samples_kp = i*N*N + j*N + np.minimum(k+1,N-1)
        next_samples_im = np.maximum(i-1,0)*N*N + j*N + k
        next_samples_jm = i*N*N + np.maximum(j-1,0)*N + k
        next_samples_km = i*N*N + j*N + np.maximum(k-1,0)

        next_indices = np.concatenate((next_samples,next_samples_ip, next_samples_jp,next_samples_kp,next_samples_im,next_samples_jm, next_samples_km))

        return verts, faces, samples, next_indices



def create_mesh_optim_fast(
    samples, indices, decoder, latent_vec, N=256, max_batch=32 ** 3, offset=None, scale=None, fourier = False, taylor = False
):

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    num_samples = indices.shape[0]

    with torch.no_grad():

        head = 0
        while head < num_samples:
            sample_subset = samples[indices[head : min(head + max_batch, num_samples)], 0:3].reshape(-1, 3).cuda()
            samples[indices[head : min(head + max_batch, num_samples)], 3] = (
                decode_sdf(decoder, latent_vec, sample_subset)
                .squeeze(1)
                .detach()
                .cpu()
            )
            head += max_batch

        sdf_values = samples[:, 3]
        sdf_values = sdf_values.reshape(N, N, N)

    verts, faces = convert_sdf_samples_to_mesh(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        offset,
        scale,
    )

    # fetch bins that are activated
    k = ((verts[:, 2] -  voxel_origin[2])/voxel_size).astype(int)
    j = ((verts[:, 1] -  voxel_origin[1])/voxel_size).astype(int)
    i = ((verts[:, 0] -  voxel_origin[0])/voxel_size).astype(int)
    # find points around
    next_samples = i*N*N + j*N + k
    next_samples_i_plus = np.minimum(i+1,N-1)*N*N + j*N + k
    next_samples_j_plus = i*N*N + np.minimum(j+1,N-1)*N + k
    next_samples_k_plus = i*N*N + j*N + np.minimum(k+1,N-1)
    next_samples_i_minus = np.maximum(i-1,0)*N*N + j*N + k
    next_samples_j_minus = i*N*N + np.maximum(j-1,0)*N + k
    next_samples_k_minus = i*N*N + j*N + np.maximum(k-1,0)
    next_indices = np.concatenate((next_samples,next_samples_i_plus, next_samples_j_plus,next_samples_k_plus,next_samples_i_minus,next_samples_j_minus, next_samples_k_minus))

    return verts, faces, samples, next_indices
