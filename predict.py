from cog import BasePredictor, Input, Path
from typing import List
import os
import glob
import requests
import urllib
import cgi
import subprocess
import torch
import safetensors
import diffusers
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from schedulers import SDXLCompatibleSchedulers

REQUESTS_GLOBAL_SESSION = requests.Session()

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

MODELS_DIR_PATH = "models"
VAES_DIR_PATH = "vaes"
MODEL_NAMES = find_models(MODELS_DIR_PATH)
assert len(MODEL_NAMES) > 0, f"You don't have any model under \"{MODELS_DIR_PATH}\", please put at least 1 model in there."
VAE_NAMES = find_vaes(VAES_DIR_PATH)
assert len(VAE_NAMES) > 0, f"You don't have any VAE under \"{VAES_DIR_PATH}\", please put at least 1 VAE in there, you can run \"python3 -c 'from huggingface_hub import snapshot_download as d;d(repo_id=\"madebyollin/sdxl-vae-fp16-fix\", allow_patterns=[\"config.json\", \"diffusion_pytorch_model.safetensors\"], local_dir=\"./vaes/sdxl-vae-fp16-fix\", local_dir_use_symlinks=False)'\" to download a fp16 fixed default SDXL VAE if you don't know what to use."
MODEL_NAMES.sort()
VAE_NAMES.sort()
TEXTUAL_INVERSION_PATHS = find_textual_inversions("textual_inversions")
SCHEDULER_NAMES = SDXLCompatibleSchedulers.get_names()

POSITIVE_PREPROMPT = "score_9, score_8_up, score_7_up, "
NEGATIVE_PREPROMPT = "score_4, score_3, score_2, score_1, worst quality, bad hands, bad feet, "

# Cog will only run this class in a single thread.
class Predictor(BasePredictor):

    def setup(self):
        self.pipelines = SDXLMultiPipelineSwitchAutoDetect(MODELS_DIR_PATH, MODEL_NAMES, VAES_DIR_PATH, VAE_NAMES, TEXTUAL_INVERSION_PATHS)
        os.makedirs("tmp", exist_ok=True)

    @torch.no_grad
    def predict(
        self,
        model: str = Input(description="The model to use", default=MODEL_NAMES[0], choices=MODEL_NAMES),
        vae: str = Input(description="The VAE to use", default=VAE_NAMES[0], choices=VAE_NAMES),
        prompt: str = Input(description="The prompt", default="1girl, cat girl, cat ears, cat tail, yellow eyes, white hair, bob cut, from side, scenery, sunset"),
        lora_url: str = Input(description="The URL to the LoRA (Will download the weights, might take a while if the LoRA is huge or the download is slow, WILL CHARGE WHEN DOWNLOADING)", default=""),
        negative_prompt: str = Input(description="The negative prompt (For things you don't want)", default="unaestheticXL_Sky3.1, animal, cat, dog, big breasts"),
        prepend_preprompt: bool = Input(description=f"Prepend preprompt (Prompt: \"{POSITIVE_PREPROMPT}\" Negative prompt: \"{NEGATIVE_PREPROMPT}\").", default=True),
        scheduler: str = Input(description="The scheduler to use", default=SCHEDULER_NAMES[0], choices=SCHEDULER_NAMES),
        steps: int = Input(description="The steps when generating", default=35, ge=1, le=100),
        cfg_scale: float = Input(description="CFG Scale defines how much attention the model pays to the prompt when generating", default=7, ge=1, le=30),
        guidance_rescale: float = Input(description="The amount to rescale CFG generated noise to avoid generating overexposed images", default=0.7, ge=0, le=1),
        width: int = Input(description="The width of the image", default=1184, ge=1, le=2048),
        height: int = Input(description="The height of the image", default=864, ge=1, le=2048),
        batch_size: int = Input(description="Number of images to generate (1-4)", default=1, ge=1, le=4),
        seed: int = Input(description="The seed used when generating, set to -1 for random seed", default=-1),
    ) -> List[Path]:
        if prompt == "__ignore__":
            return []
        pipeline = self.pipelines.get_pipeline(model, vae, scheduler)
        pipeline.unload_lora_weights()
        if lora_url:
            process_lora(lora_url, pipeline)
        generator = None
        if seed != -1:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        if prepend_preprompt:
            prompt = POSITIVE_PREPROMPT + prompt
            negative_prompt = NEGATIVE_PREPROMPT + negative_prompt
        imgs = pipeline(
            prompt=prompt, negative_prompt=negative_prompt, width=width, height=height, num_inference_steps=steps,
            guidance_scale=cfg_scale, guidance_rescale=guidance_rescale, num_images_per_prompt=batch_size, generator=generator,
        ).images

        image_paths = []
        for index, img in enumerate(imgs):
            img_file_path = f"tmp/{index}.png"
            img.save(img_file_path, compression=9)
            image_paths.append(Path(img_file_path))

        return image_paths

def check_url(url):
    with REQUESTS_GLOBAL_SESSION.get(url, allow_redirects=True, timeout=5, stream=True) as response:
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

LORAS_DIR_PATH = "loras"
URL_LORA_FILENAME_DICT = {}

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

class SDXLMultiPipelineSwitchAutoDetect:

    def __init__(self, models_dir_path, model_names, vaes_dir_path, vae_names, textual_inversion_paths):
        self.models_dir_path = models_dir_path
        self.model_pipeline_dict = {model_name: None for model_name in model_names} # Key = Model's name(str), Value = StableDiffusionXLPipeline instance
        self.vaes_dir_path = vaes_dir_path
        self.vae_obj_dict = {vae_name: None for vae_name in vae_names} # Key = VAE's name(str), Value = AutoencoderKL instance
        self.textual_inversion_paths = textual_inversion_paths

        self._load_all_models()
        self._load_all_vaes()

        self.on_cuda_model = model_names[0]
        on_cuda_pipeline = self.model_pipeline_dict[self.on_cuda_model]
        on_cuda_pipeline.to("cuda")
        on_cuda_pipeline.vae = self.vae_obj_dict[vae_names[0]]

    def get_pipeline(self, model_name, vae_name, scheduler_name):
        pipeline = self.model_pipeline_dict.get(model_name)
        vae = self.vae_obj_dict.get(vae_name)
        # __init__ function guarantees models and VAEs to be loaded.
        if pipeline is None:
            raise ValueError(f"Model \"{model_name}\" not found.")
        if vae is None:
            raise ValueError(f"VAE \"{vae_name}\" not found.")

        if model_name != self.on_cuda_model:
            prev_on_cuda_pipeline = self.model_pipeline_dict[self.on_cuda_model]
            prev_on_cuda_pipeline.vae = None
            prev_on_cuda_pipeline.to("cpu")
            pipeline.to("cuda")
            self.on_cuda_model = model_name

        pipeline.vae = vae
        pipeline.scheduler = SDXLCompatibleSchedulers.create_instance(scheduler_name)
        return pipeline

    # Load all models to CPU.
    def _load_all_models(self):
        clip_l_list, clip_g_list, activation_token_list = get_textual_inversions(self.textual_inversion_paths)
        for model_name in self.model_pipeline_dict.keys():
            self.model_pipeline_dict[model_name] = self._load_model(model_name, clip_l_list, clip_g_list, activation_token_list)

    # Load a model to CPU.
    def _load_model(self, model_name, clip_l_list, clip_g_list, activation_token_list):
        pipeline = StableDiffusionXLPipeline.from_single_file(os.path.join(self.models_dir_path, model_name), torch_dtype=torch.bfloat16, variant="fp16", add_watermarker=False)
        apply_textual_inversions_to_sdxl_pipeline(pipeline, clip_l_list, clip_g_list, activation_token_list)
        pipeline.vae = None
        pipeline.enable_xformers_memory_efficient_attention()
        return pipeline

    # Load all VAEs to GPU(CUDA).
    def _load_all_vaes(self):
        for vae_name in self.vae_obj_dict.keys():
            self.vae_obj_dict[vae_name] = self._load_vae(vae_name)

    # Load a VAE to GPU(CUDA).
    def _load_vae(self, vae_name):
        vae = AutoencoderKL.from_pretrained(os.path.join(self.vaes_dir_path, vae_name), torch_dtype=torch.bfloat16)
        vae.enable_slicing()
        vae.enable_tiling()
        vae.to("cuda")
        return vae

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
