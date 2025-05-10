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
MODELS_REMOTE_CACHE_PATH = "__models_remote_cache__"
import finders # finders.py | Need to import finders here because MODELS_REMOTE_CACHE_PATH must be set before model_for_loading.py imports constants.py (this script).
MODELS = finders.find_models(MODELS_DIR_PATH)
MODEL_NAMES = list(MODELS)

VAES_DIR_PATH = "vaes"
VAE_NAMES = finders.find_vaes(VAES_DIR_PATH)
VAE_NAMES.sort()

LORAS_DIR_PATH = "loras"
MAX_LORA_CACHE_BYTES = 34359738368 # 32 GB.

TEXTUAL_INVERSION_PATHS = finders.find_textual_inversions("textual_inversions")

SCHEDULER_NAMES = SDXLCompatibleSchedulers.get_names()

DEFAULT_MODEL = MODEL_NAMES[0] if len(MODEL_NAMES) else None

DEFAULT_VAE_NAME = None
BAKEDIN_VAE_LABEL = "default"

DEFAULT_LORA = None

DEFAULT_POS_PREPROMPT = "masterpiece, best quality, absurdres, "
DEFAULT_NEG_PREPROMPT = "worst quality, english text, japanese text, twitter username, watermark, (bad feet)1.5, jpeg artifacts, "

DEFAULT_POSITIVE_PROMPT = "1girl"
DEFAULT_NEGATIVE_PROMPT = "animal, cat, dog, big breasts"

DEFAULT_CFG = 5
DEFAULT_RESCALE = 0.5
DEFAULT_PAG = 3

# CLIP skip is calculated in the same way as AUTOMATIC1111 and CivitAI.
DEFAULT_CLIP_SKIP = 1

DEFAULT_WIDTH = 1184
DEFAULT_HEIGHT = 864

DEFAULT_SCHEDULER = SCHEDULER_NAMES[0]
DEFAULT_STEPS = 35
