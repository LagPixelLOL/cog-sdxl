import os
import cgi
import torch
import urllib
import subprocess
import safetensors
from PIL import Image
from constants import * # constants.py

def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong
    # background for transparent images. The call to `alpha_composite` handles this case.
    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite

def scale_and_crop(image_path, width, height):
    with Image.open(image_path) as img:
        img = convert_to_rgb(img)

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

def get_download_info(url, cookies=None):
    with REQUESTS_GLOBAL_SESSION.get(url, cookies=cookies, timeout=5, stream=True) as response:
        if response.status_code >= 200 and response.status_code < 300:
            content_disposition = response.headers.get("Content-Disposition")
            if content_disposition:
                value, params = cgi.parse_header(content_disposition)
                if "filename*" in params:
                    encoding, _, filename = params["filename*"].split("'", 2)
                    return response.url, urllib.parse.unquote(filename, encoding=encoding)
                elif "filename" in params:
                    return response.url, urllib.parse.unquote(params["filename"])
            content_type = response.headers.get("Content-Type")
            if content_type == "binary/octet-stream":
                parsed_url = urllib.parse.urlparse(url)
                filename = parsed_url.path.rsplit("/", 1)[-1]
                if filename:
                    return response.url, urllib.parse.unquote(filename)
        else:
            raise RuntimeError("URL responded with status code: " + str(response.status_code))
    raise RuntimeError("URL not downloadable.")

def download(url, destination):
    return_code = subprocess.run(["pget", "-m", "10M", "-c", "320", url, destination]).returncode
    if return_code != 0:
        raise RuntimeError(f"Failed to download \"{filename}\", return code: {return_code}")

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
