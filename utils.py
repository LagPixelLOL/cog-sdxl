import os
import glob
import requests
import urllib
import cgi
import subprocess
import torch
import safetensors
from PIL import Image

REQUESTS_UTILS_SESSION = requests.Session()

LORAS_DIR_PATH = "loras"
URL_LORA_FILENAME_DICT = {}

def scale_and_crop(image_path, width, height):
    img = Image.open(image_path).convert('RGB')

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

    img_resized = img.resize((new_width, new_height), Image.LANCZOS)

    left = (new_width - width) // 2
    top = (new_height - height) // 2
    right = left + width
    bottom = top + height

    img_cropped = img_resized.crop((left, top, right, bottom))

    return img_cropped

def process_lora(url, pipeline):
    url = url.strip()
    if url not in URL_LORA_FILENAME_DICT:
        filename = check_url(url)
        _, ext = os.path.splitext(filename)
        if ext not in [".safetensors", ".bin"]:
            raise RuntimeError("URL file extension not supported:", ext)
        lora_path = os.path.join(LORAS_DIR_PATH, filename)
        if filename not in URL_LORA_FILENAME_DICT.values():
            return_code = subprocess.run(["pget", "-m", "10M", url, lora_path]).returncode
            if return_code != 0:
                raise RuntimeError(f"Failed to download \"{filename}\", return code:", return_code)
        URL_LORA_FILENAME_DICT[url] = filename
    else:
        filename = URL_LORA_FILENAME_DICT[url]
        _, ext = os.path.splitext(filename)
        lora_path = os.path.join(LORAS_DIR_PATH, filename)

    if ext == ".safetensors":
        state_dict = safetensors.torch.load_file(lora_path)
    else:
        state_dict = torch.load(lora_path)
    pipeline.load_lora_weights(state_dict, filename.replace(".", "_"))

def check_url(url):
    with REQUESTS_UTILS_SESSION.get(url, allow_redirects=True, timeout=5, stream=True) as response:
        if response.status_code >= 200 and response.status_code < 300:
            content_disposition = response.headers.get("Content-Disposition")
            if content_disposition:
                value, params = cgi.parse_header(content_disposition)
                if "filename*" in params:
                    encoding, _, filename = params["filename*"].split("'", 2)
                    return urllib.parse.unquote(filename, encoding=encoding)
                elif "filename" in params:
                    return urllib.parse.unquote(params["filename"])
            content_type = response.headers.get("Content-Type")
            if content_type == "binary/octet-stream":
                parsed_url = urllib.parse.urlparse(url)
                filename = parsed_url.path.split("/")[-1]
                if filename:
                    return urllib.parse.unquote(filename)
        else:
            raise RuntimeError(f"URL responded with status code: {response.status_code}")
    raise RuntimeError("URL not downloadable.")

# Returns the base filenames of the models.
def find_models(models_dir):
    return [os.path.basename(file) for file in glob.glob(f"{models_dir}/**/*.safetensors", recursive=True) + glob.glob(f"{models_dir}/**/*.ckpt", recursive=True)]

# Returns the folder names of the VAEs.
def find_vaes(vaes_dir):
    vae_names = []
    for folder in os.listdir(vaes_dir):
        folder_path = os.path.join(vaes_dir, folder)
        if os.path.isdir(folder_path):
            safetensors_file = os.path.join(folder_path, 'diffusion_pytorch_model.safetensors')
            bin_file = os.path.join(folder_path, 'diffusion_pytorch_model.bin')
            config_file = os.path.join(folder_path, 'config.json')
            if (os.path.isfile(safetensors_file) or os.path.isfile(bin_file)) and os.path.isfile(config_file):
                vae_names.append(folder)
    return vae_names

# Returns the relative paths of the textual inversions.
def find_textual_inversions(textual_inversions_dir):
    return [file for file in glob.glob(f"{textual_inversions_dir}/**/*.safetensors", recursive=True) + glob.glob(f"{textual_inversions_dir}/**/*.bin", recursive=True)]

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
