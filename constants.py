import finders # finders.py
import os
import requests
import torch

TORCH_DTYPE = torch.bfloat16

MODELS_DIR_PATH = "models"
MODELS = finders.find_models(MODELS_DIR_PATH)

VAES_DIR_PATH = "vaes"
VAE_NAMES = finders.find_vaes(VAES_DIR_PATH)
VAE_NAMES.sort()

DEFAULT_VAE_NAME = "default"

LORAS_DIR_PATH = "loras"

TEXTUAL_INVERSION_PATHS = finders.find_textual_inversions("textual_inversions")

REQUESTS_GLOBAL_SESSION = requests.Session()

POSITIVE_PREPROMPT = "score_9, score_8_up, score_7_up, "
NEGATIVE_PREPROMPT = "score_4, score_3, score_2, score_1, worst quality, bad hands, bad feet, "
