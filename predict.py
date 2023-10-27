from cog import BasePredictor, Input, Path
from typing import List
import os
import glob
import torch
import diffusers
from diffusers import StableDiffusionXLPipeline
from PIL import Image

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

MODELS_DIR_PATH = "models"
MODEL_NAMES = find_models(MODELS_DIR_PATH)
assert(len(MODEL_NAMES) > 0)

class Predictor(BasePredictor):

    def setup(self):
        self.pipelines = SDXLMultiPipelineSwitchAutoDetect(MODELS_DIR_PATH, MODEL_NAMES)
        if not os.path.exists("tmp"):
            os.makedirs("tmp")

    @torch.inference_mode
    def predict(
        self,
        model: str = Input(description="The model to use", default=MODEL_NAMES[0], choices=MODEL_NAMES),
        prompt: str = Input(description="The prompt", default="catgirl, cat ears, white hair, golden eyes, bob cut, pov, face closeup, smile"),
        negative_prompt: str = Input(description="The negative prompt (For things you don't want)", default="lowres, low quality, worse quality, monochrome, blurry, headphone, big breasts"),
        steps: int = Input(description="The steps when generating", default=35, ge=1, le=100),
        cfg_scale: float = Input(description="CFG Scale defines how much attention the model pays to the prompt when generating", default=7, ge=1, le=30),
        guidance_rescale: float = Input(description="The amount to rescale CFG generated noise to avoid generating overexposed images", default=0.7, ge=0, le=1),
        width: int = Input(description="The width of the image", default=1184, ge=1, le=2048),
        height: int = Input(description="The height of the image", default=864, ge=1, le=2048),
        batch_size: int = Input(description="Number of images to generate (1-4)", default=1, ge=1, le=4),
        seed: int = Input(description="The seed used when generating, set to -1 for random seed", default=-1),
    ) -> List[Path]:
        pipeline = self.pipelines.get_pipeline(model)
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

    def __init__(self, models_dir_path, model_names):
        self.models_dir_path = models_dir_path
        self.model_pipeline_dict = {model: None for model in model_names}
        self.vae = diffusers.AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.bfloat16)
        self.vae.enable_slicing()
        self.vae.enable_tiling()
        self._load_all_models()
        self.on_cuda_model = model_names[0]
        self.model_pipeline_dict[self.on_cuda_model].to("cuda")

    def get_pipeline(self, model_name):
        pipeline = self.model_pipeline_dict.get(model_name)
        if pipeline is None:
            raise ValueError(f"Model '{model_name}' not found or not loaded.")

        if model_name != self.on_cuda_model:
            self.model_pipeline_dict[self.on_cuda_model].to("cpu")
            pipeline.to("cuda")
            self.on_cuda_model = model_name

        return pipeline

    def _load_all_models(self):
        for model_name in self.model_pipeline_dict.keys():
            pipeline = self._load_model(model_name)
            self.model_pipeline_dict[model_name] = pipeline

    def _load_model(self, model_name):
        model_path = os.path.join(self.models_dir_path, model_name)
        pipeline = StableDiffusionXLPipeline.from_single_file(
            model_path, vae=self.vae, torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True, add_watermarker=False,
        )
        pipeline.scheduler = diffusers.UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline.enable_xformers_memory_efficient_attention()
        return pipeline
