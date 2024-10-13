from constants import * # constants.py
SINGLE_FILE_MODELS = {model_name: model for model_name, model in MODELS.items() if model.is_single_file and not model.is_remote}
assert len(SINGLE_FILE_MODELS) > 0, f"You don't have any single file local model under \"{MODELS_DIR_PATH}\", please put at least 1 single file local model in there."

import os
import argparse
from diffusers import StableDiffusionXLPipeline

def parse_args():
    choices = [model_name for model_name in SINGLE_FILE_MODELS]
    model_name_required = len(choices) > 1
    parser = argparse.ArgumentParser(description="Extract Stable Diffusion XL single file local model into HuggingFace Diffusers format.")
    parser.add_argument(
        "-o",
        "--output-name",
        help="Name for the extracted model",
    )
    parser.add_argument(
        "-n",
        "--no-delete",
        action="store_true",
        help="Disable auto deletion of the old model",
    )
    parser.add_argument(
        "model_name",
        nargs=None if model_name_required else "?",
        choices=choices,
        default=None if model_name_required else choices[0],
        help="Name of the model to extract from",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    model = SINGLE_FILE_MODELS[args.model_name]
    model_path = model.model_uri
    print(f"Loading single file local model from \"{model_path}\"...")
    pipeline = model.load(torch_dtype=TORCH_DTYPE, add_watermarker=False)
    out_model_path = os.path.join(MODELS_DIR_PATH, args.output_name if args.output_name else args.model_name)
    print(f"Saving HuggingFace Diffusers format model to \"{out_model_path}\"...")
    pipeline.save_pretrained(out_model_path)
    if not args.no_delete:
        print(f"Deleting the old model \"{model_path}\"...")
        os.remove(model_path)
    print("Finished.")

if __name__ == "__main__":
    main()
