from constants import * # constants.py
assert len(MODEL_NAMES) > 0, f"You don't have any model under \"{MODELS_DIR_PATH}\", please put at least 1 model in there."

import os
import shutil
import argparse
from diffusers import StableDiffusionXLPipeline

def parse_args():
    vae_name_required = DEFAULT_VAE_NAME in VAE_NAMES
    model_name_required = len(MODEL_NAMES) > 1
    parser = argparse.ArgumentParser(description="Extract VAE from Stable Diffusion XL model")
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        required=model_name_required,
        choices=MODEL_NAMES,
        default=MODEL_NAMES[0] if not model_name_required else None,
        help="Name of the model to extract VAE from"
    )
    parser.add_argument(
        "-v",
        "--vae-name",
        type=str,
        required=vae_name_required,
        default=DEFAULT_VAE_NAME if not vae_name_required else None,
        help="Name for the extracted VAE"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = os.path.join(MODELS_DIR_PATH, args.model_name)
    print(f"Loading model from \"{model_path}\"...")
    pipeline = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=TORCH_DTYPE, add_watermarker=False)
    vae_path = os.path.join(VAES_DIR_PATH, args.vae_name)
    print(f"Saving VAE to \"{vae_path}\"...")
    pipeline.vae.save_pretrained(vae_path)
    pycache_dir = "__pycache__"
    if os.path.exists(pycache_dir):
        shutil.rmtree(pycache_dir)

if __name__ == "__main__":
    main()
