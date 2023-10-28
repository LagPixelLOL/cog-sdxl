# predict.py

# Internals
from cog import BasePredictor, Input, Path
from typing import List
import os
import torch

# Externals
from modules.sdxl_pipeline import SDXLMultiPipelineSwitchAutoDetect
from modules.helpers import MODEL_NAMES, VAE_NAMES
from constants.directories import MODELS_DIR_PATH, VAE_DIR_PATH
from constants.schedulers import SCHEDULERS

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