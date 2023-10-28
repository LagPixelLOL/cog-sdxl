from cog import BasePredictor, Input, Path
import diffusers
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, UniPCMultistepScheduler
from typing import List
import os
import glob
import torch

def find_models(models_dir):
    model_names = []
    safetensor_files = glob.glob(f"{models_dir}/**/*.safetensors", recursive=True)
    ckpt_files = glob.glob(f"{models_dir}/**/*.ckpt", recursive=True)

    for file in safetensor_files:
        if not file.endswith(".vae.safetensors"):
            model_names.append(os.path.basename(file))
    for file in ckpt_files:
        if not file.endswith(".vae.ckpt"):
            model_names.append(os.path.basename(file))

    return model_names

def find_vae_models(vae_dir):
    vae_names = []
    safetensor_files = glob.glob(f"{vae_dir}/**/*.safetensors", recursive=True)
    ckpt_files = glob.glob(f"{vae_dir}/**/*.ckpt", recursive=True)

    for file in safetensor_files:
        vae_names.append(os.path.basename(file))
    for file in ckpt_files:
        vae_names.append(os.path.basename(file))

    return vae_names

MODELS_DIR_PATH = "models"
VAE_DIR_PATH = "vae"

MODEL_NAMES = find_models(MODELS_DIR_PATH)
VAE_NAMES = find_vae_models(VAE_DIR_PATH)

SCHEDULERS = [
    'DDIMScheduler',
    'LMSDiscreteScheduler',
    'PNDMScheduler',
    'UniPCMultistepScheduler'
]

assert(len(MODEL_NAMES) > 0)
assert(len(VAE_NAMES) > 0)

class Predictor(BasePredictor):

    def setup(self):
        self.pipelines = SDXLMultiPipelineSwitchAutoDetect(MODELS_DIR_PATH, VAE_DIR_PATH, MODEL_NAMES, VAE_NAMES)
        if not os.path.exists("tmp"):
            os.makedirs("tmp")

    @torch.inference_mode
    def predict(
        self,
        model: str = Input(description="The list of models you can use", default=MODEL_NAMES[0], choices=MODEL_NAMES),
        prompt: str = Input(description="The prompt in which the model will try to create", default="masterpiece, illustration, self-portrait of a young woman with golden blonde hair and pale blue eyes wearing an intricate white dress"),
        negative_prompt: str = Input(description="The prompt in which the model will try to avoid", default="low quality, worst quality, bad quality"),
        steps: int = Input(description="The number of steps it tries to create the image", default=20, ge=1, le=100),
        cfg_scale: float = Input(description="CFG Scale defines how much attention the model pays to the prompt when generating", default=7, ge=1, le=30),
        guidance_rescale: float = Input(description="The amount to rescale CFG generated noise to avoid generating overexposed images", default=0.7, ge=0, le=1),
        width: int = Input(description="The width of the image", default=1184, ge=1, le=2048),
        height: int = Input(description="The height of the image", default=864, ge=1, le=2048),
        vae: str = Input(description="The VAE model to use", default=VAE_NAMES[0], choices=VAE_NAMES),
        scheduler: str = Input(description="The scheduler to use", default=SCHEDULERS[0], choices=SCHEDULERS),
        batch_size: int = Input(description="Number of images to generate (1-4)", default=1, ge=1, le=4),
        seed: int = Input(description="The seed used when generating, set to -1 for random seed", default=-1),
    ) -> List[Path]:
        pipeline = self.pipelines.get_pipeline(model, vae, scheduler)
        generator = None
        if seed != -1:
            generator = torch.Generator(device="cuda").manual_seed(seed)
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

class SDXLMultiPipelineSwitchAutoDetect():

    def __init__(self, models_dir_path, vae_dir_path, model_names, vae_names):
        self.models_dir_path = models_dir_path
        self.vae_dir_path = vae_dir_path
        self.model_pipeline_dict = {model: {vae: {scheduler: None for scheduler in SCHEDULERS} for vae in vae_names} for model in model_names}
        self.on_cuda_model = model_names[0]
        self.on_cuda_vae = vae_names[0]
        self.on_cuda_scheduler = SCHEDULERS[0]
        self._load_all_models()
        self.model_pipeline_dict[self.on_cuda_model][self.on_cuda_vae][self.on_cuda_scheduler].to("cuda")
        
    def get_pipeline(self, model_name, vae_name, scheduler_name):
        pipeline = self.model_pipeline_dict.get(model_name).get(vae_name).get(scheduler_name)
        if pipeline is None:
            raise ValueError(f"Model '{model_name}', VAE model '{vae_name}' or scheduler '{scheduler_name}' not found or not loaded.")

        if model_name != self.on_cuda_model or vae_name != self.on_cuda_vae or scheduler_name != self.on_cuda_scheduler:
            self.model_pipeline_dict[self.on_cuda_model][self.on_cuda_vae][self.on_cuda_scheduler].to("cpu")
            pipeline.to("cuda")
            self.on_cuda_model = model_name
            self.on_cuda_vae = vae_name
            self.on_cuda_scheduler = scheduler_name

        return pipeline

    def _load_all_models(self):
        for model_name in self.model_pipeline_dict.keys():
            for vae_name in self.model_pipeline_dict[model_name]:
                for scheduler_name in SCHEDULERS:
                    pipeline = self._load_model(model_name, vae_name, scheduler_name)
                    self.model_pipeline_dict[model_name][vae_name][scheduler_name] = pipeline

    def _load_model(self, model_name, vae_name, scheduler_name):
        model_path = os.path.join(self.models_dir_path, model_name)
        vae_path = os.path.join(self.vae_dir_path, vae_name)
        vae = diffusers.AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.bfloat16)          
        pipeline = StableDiffusionXLPipeline.from_single_file(
            model_path, vae=vae, torch_dtype=torch.bfloat16, variant="fp16", add_watermarker=False,
        )
        pipeline.scheduler = globals()[scheduler_name].from_config(pipeline.scheduler.config)
        pipeline.enable_xformers_memory_efficient_attention()
        return pipeline
