import finders # finders.py
import requests
import torch
from schedulers import SDXLCompatibleSchedulers # schedulers.py

TORCH_DTYPE = torch.bfloat16
REQUESTS_GLOBAL_SESSION = requests.Session()

# When set to True, the script will offload inactive models to CPU, this will add 3 seconds of overhead when switching models.
# When set to False, the script won't offload inactive models to CPU, all models will be in GPU which means they will use VRAM,
# but it should make switching models way faster because the script won't need to move the models between RAM and VRAM.
CPU_OFFLOAD_INACTIVE_MODELS = False

MODELS_DIR_PATH = "models"
MODELS = finders.find_models(MODELS_DIR_PATH)
MODEL_NAMES = list(MODELS)

VAES_DIR_PATH = "vaes"
VAE_NAMES = finders.find_vaes(VAES_DIR_PATH)
VAE_NAMES.sort()

LORAS_DIR_PATH = "loras"
MAX_LORA_CACHE_BYTES = 137438953472 # 128 GB.

TEXTUAL_INVERSION_PATHS = finders.find_textual_inversions("textual_inversions")

SCHEDULER_NAMES = SDXLCompatibleSchedulers.get_names()


DEFAULT_MODEL = MODEL_NAMES[0]

DEFAULT_VAE_NAME = None
BAKEDIN_VAE_LABEL = "default"

DEFAULT_LORA = None

DEFAULT_POS_PREPROMPT = "score_9, score_8_up, score_7_up, "
DEFAULT_NEG_PREPROMPT = "score_4, score_3, score_2, score_1, worst quality, bad hands, bad feet, "

DEFAULT_POSITIVE_PROMPT = "safe"
DEFAULT_NEGATIVE_PROMPT = ""

DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024

DEFAULT_STEPS = 35
DEFAULT_SCHEDULER = SCHEDULER_NAMES[0]

DEFAULT_CFG = 7
DEFAULT_GUIDANCE = 0.7

# Clip Skip is calculated in the same way AUTOMATIC1111 or CivitAI
DEFAULT_CLIP_SKIP = 1
