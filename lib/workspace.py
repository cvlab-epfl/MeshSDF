#!/usr/bin/env python3

import json
import os
import torch
import pdb

model_params_subdir = "ModelParameters"
optimizer_params_subdir = "OptimizerParameters"
latent_codes_subdir = "LatentCodes"
logs_filename = "Logs.pth"
reconstructions_subdir = "Reconstructions"
optimizations_subdir = "Optimizations"
optimizations_meshes_subdir = "Meshes"
optimizations_codes_subdir = "Codes"
specifications_filename = "specs.json"
data_source_map_filename = ".datasources.json"


def load_experiment_specifications(experiment_directory):

    filename = os.path.join(experiment_directory, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )

    return json.load(open(filename))

def load_cam_model_parameters(experiment_directory, checkpoint, encoder):

    filename = os.path.join(
        experiment_directory, model_params_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename)

    encoder.load_state_dict(data["model_state_dict"])

    return data["epoch"]

def load_model_parameters(experiment_directory, checkpoint, decoder):

    filename = os.path.join(
        experiment_directory, model_params_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename)

    decoder.load_state_dict(data["model_state_dict"])

    return data["epoch"]


def load_model_parameters(experiment_directory, checkpoint, encoder, decoder):

    filename = os.path.join(
        experiment_directory, model_params_subdir, "encoder_" + checkpoint + ".pth"
    )
    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename)
    encoder.load_state_dict(data["model_state_dict"])

    filename = os.path.join(
        experiment_directory, model_params_subdir, "decoder_" + checkpoint + ".pth"
    )
    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename)
    decoder.load_state_dict(data["model_state_dict"])

    return data["epoch"]




def build_decoder(experiment_directory, experiment_specs):

    arch = __import__(
        "networks." + experiment_specs["NetworkArch"], fromlist=["Decoder"]
    )

    latent_size = experiment_specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **experiment_specs["NetworkSpecs"]).cuda()

    return decoder


def load_decoder(
    experiment_directory, experiment_specs, checkpoint, data_parallel=True
):

    decoder = build_decoder(experiment_directory, experiment_specs)

    if data_parallel:
        decoder = torch.nn.DataParallel(decoder)

    epoch = load_model_parameters(experiment_directory, checkpoint, decoder)

    return (decoder, epoch)


def load_latent_vectors(experiment_directory, checkpoint):

    filename = os.path.join(
        experiment_directory, latent_codes_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include a latent code file"
            + " for checkpoint '{}'".format(experiment_directory, checkpoint)
        )

    data = torch.load(filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        num_vecs = data["latent_codes"].size()[0]

        lat_vecs = []
        for i in range(num_vecs):
            lat_vecs.append(data["latent_codes"][i].cuda())

        return lat_vecs

    else:

        num_embeddings, embedding_dim = data["latent_codes"]["weight"].shape

        lat_vecs = torch.nn.Embedding(num_embeddings, embedding_dim)

        lat_vecs.load_state_dict(data["latent_codes"])

        return lat_vecs.weight.data.detach()


def get_data_source_map_filename(data_dir):
    return os.path.join(data_dir, data_source_map_filename)


def get_reconstructed_code_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_codes_subdir,
        dataset,
        class_name,
        instance_name + ".pth",
    )


def get_evaluation_dir(experiment_dir, checkpoint, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, evaluation_subdir, checkpoint)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_model_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, model_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, optimizer_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_latent_codes_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, latent_codes_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir

def get_reconstruction_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, reconstructions_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir

def get_mesh_filename(reconstruction_dir, name):
    dir = os.path.join(reconstruction_dir, name.split("/")[-1].replace(".npz", ""))
    return dir


def get_normalization_params_filename(
    data_dir, dataset_name, class_name, instance_name
):
    return os.path.join(
        data_dir,
        normalization_param_subdir,
        dataset_name,
        class_name,
        instance_name + ".npz",
    )
