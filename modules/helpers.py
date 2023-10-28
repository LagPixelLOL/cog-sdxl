# modules/helpers.py

import glob
import os
from constants.directories import MODELS_DIR_PATH, VAE_DIR_PATH


def find_models(models_dir):
    model_names = []
    safetensor_files = glob.glob(f"{models_dir}/**/*.safetensors", recursive=True)
    ckpt_files = glob.glob(f"{models_dir}/**/*.ckpt", recursive=True)

    for file in safetensor_files:
        if not file.endswith(".vae.safetensors"):
            model_names.append(os.path.basename(file))
    for file in ckpt_files:
        if not file.endswith(".vae.ckpt"):
            model_names.append(os.path.basename(file))

    return model_names


def find_vae_models(vae_dir):
    vae_names = []
    safetensor_files = glob.glob(f"{vae_dir}/**/*.safetensors", recursive=True)
    ckpt_files = glob.glob(f"{vae_dir}/**/*.ckpt", recursive=True)

    for file in safetensor_files:
        vae_names.append(os.path.basename(file))
    for file in ckpt_files:
        vae_names.append(os.path.basename(file))

    return vae_names


MODEL_NAMES = find_models(MODELS_DIR_PATH)

VAE_NAMES = find_vae_models(VAE_DIR_PATH)