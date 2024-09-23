import finders # finders.py
import os
import requests
import torch

TORCH_DTYPE = torch.bfloat16

MODELS_DIR_PATH = "models"
MODELS = finders.find_models(MODELS_DIR_PATH)

# When set to True, the script will offload inactive models to CPU, this will add 3 seconds of overhead when switching models.
# When set to False, the script won't offload inactive models to CPU, all models will be in GPU which means they will use VRAM,
# but it should make switching models way faster because the script won't need to move the models between RAM and VRAM.
CPU_OFFLOAD_INACTIVE_MODELS = False

VAES_DIR_PATH = "vaes"
VAE_NAMES = finders.find_vaes(VAES_DIR_PATH)
VAE_NAMES.sort()

DEFAULT_VAE_NAME = None
DEFAULT_DEFAULT_VAE_NAME = "default"
DEFAULT_VAE_NAME = DEFAULT_DEFAULT_VAE_NAME if DEFAULT_VAE_NAME is None else DEFAULT_VAE_NAME

LORAS_DIR_PATH = "loras"
MAX_LORA_CACHE_BYTES = 137438953472 # 128 GB.

TEXTUAL_INVERSION_PATHS = finders.find_textual_inversions("textual_inversions")

REQUESTS_GLOBAL_SESSION = requests.Session()

POSITIVE_PREPROMPT = "score_9, score_8_up, score_7_up, "
NEGATIVE_PREPROMPT = "score_4, score_3, score_2, score_1, worst quality, bad hands, bad feet, "
