from constants import * # constants.py
SINGLE_FILE_MODELS = {model_name: model for model_name, model in MODELS.items() if model.is_single_file}
assert len(SINGLE_FILE_MODELS) > 0, f"You don't have any single file model under \"{MODELS_DIR_PATH}\", please put at least 1 single file model in there."

import os
import shutil
import argparse
from typing import Optional
from diffusers import StableDiffusionXLPipeline

def parse_args():
    choices = [model_name for model_name in SINGLE_FILE_MODELS]
    model_name_required = len(choices) > 1
    parser = argparse.ArgumentParser(description="Extract Stable Diffusion XL sigle file model into HuggingFace Diffusers format.")
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        required=model_name_required,
        choices=choices,
        default=choices[0] if not model_name_required else None,
        help="Name of the model to extract from.",
    )
    parser.add_argument(
        "-o",
        "--output-name",
        type=Optional[str],
        help="Name for the extracted model.",
    )
    parser.add_argument(
        "-n",
        "--no-delete",
        action="store_true",
        help="Disable auto deletion of the old model.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = SINGLE_FILE_MODELS[args.model_name].model_path
    print(f"Loading single file model from \"{model_path}\"...")
    pipeline = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=TORCH_DTYPE, add_watermarker=False)
    out_model_path = os.path.join(MODELS_DIR_PATH, args.output_name if args.output_name else args.model_name)
    print(f"Saving HuggingFace Diffusers format model to \"{out_model_path}\"...")
    pipeline.save_pretrained(out_model_path)
    if not args.no_delete:
        print(f"Deleting the old model \"{model_path}\"...")
        os.remove(model_path)
    shutil.rmtree("__pycache__", ignore_errors=True)
    print("Finished.")

if __name__ == "__main__":
    main()
