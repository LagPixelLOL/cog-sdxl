import os
import torch
import safetensors
from PIL import Image

def scale_and_crop(image_path, width, height):
    img = Image.open(image_path)

    img_ratio = img.width / img.height
    target_ratio = width / height

    if img_ratio > target_ratio:
        scale_factor = height / img.height
        new_width = int(img.width * scale_factor)
        new_height = height
    else:
        scale_factor = width / img.width
        new_width = width
        new_height = int(img.height * scale_factor)

    img_resized = img.resize((new_width, new_height))

    left = (new_width - width) // 2
    top = (new_height - height) // 2
    right = left + width
    bottom = top + height

    return img_resized.crop((left, top, right, bottom))

def apply_textual_inversions_to_sdxl_pipeline(sdxl_pipeline, clip_l_list, clip_g_list, activation_token_list):
    if clip_l_list and clip_g_list and activation_token_list:
        sdxl_pipeline.load_textual_inversion(clip_l_list, activation_token_list, sdxl_pipeline.tokenizer, sdxl_pipeline.text_encoder)
        sdxl_pipeline.load_textual_inversion(clip_g_list, activation_token_list, sdxl_pipeline.tokenizer_2, sdxl_pipeline.text_encoder_2)

def get_textual_inversions(textual_inversion_paths):
    clip_l_list = []
    clip_g_list = []
    activation_token_list = []
    for textual_inversion_path in textual_inversion_paths:
        filename, ext = os.path.splitext(os.path.basename(textual_inversion_path))
        if ext == ".safetensors":
            state_dict = safetensors.torch.load_file(textual_inversion_path)
        else:
            state_dict = torch.load(textual_inversion_path)
        clip_l_list.append(state_dict['clip_l'])
        clip_g_list.append(state_dict['clip_g'])
        activation_token_list.append(filename)
    return (clip_l_list, clip_g_list, activation_token_list)
