from constants import * # constants.py
assert len(MODELS) > 0, f"You don't have any model under \"{MODELS_DIR_PATH}\", please put at least 1 model in there."

from cog import BasePredictor, Input, Path
import finders # finders.py
import utils # utils.py
import os
import random
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, AutoencoderKL
from schedulers import SDXLCompatibleSchedulers # schedulers.py

MODEL_NAMES = list(MODELS)
SCHEDULER_NAMES = SDXLCompatibleSchedulers.get_names()

# Cog will only run this class in a single thread.
class Predictor(BasePredictor):

    def setup(self):
        os.environ.update({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
        self.pipelines = SDXLMultiPipelineSwitchAutoDetect(MODELS, VAES_DIR_PATH, VAE_NAMES, TEXTUAL_INVERSION_PATHS)
        os.makedirs("tmp", exist_ok=True)

    def predict(
        self,
        model: str = Input(description="The model to use", default=MODEL_NAMES[0], choices=MODEL_NAMES),
        vae: str = Input(description="The VAE to use", default=DEFAULT_VAE_NAME, choices=[DEFAULT_VAE_NAME] + VAE_NAMES + MODEL_NAMES),
        prompt: str = Input(description="The prompt", default="1girl"),
        image: Path = Input(description="The image for image to image or as the base for inpainting (Will be scaled then cropped to the set width and height)", default=None),
        mask: Path = Input(description="The mask for inpainting, white areas will be modified and black preserved (Will be scaled then cropped to the set width and height)", default=None),
        lora_url: str = Input(description="The URL to the LoRA (Will download the weights, might take a while if the LoRA is huge or the download is slow, WILL CHARGE WHEN DOWNLOADING)", default=None),
        negative_prompt: str = Input(description="The negative prompt (For things you don't want)", default="animal, cat, dog, big breasts"),
        prepend_preprompt: bool = Input(description=f"Prepend preprompt (Prompt: \"{POSITIVE_PREPROMPT}\" Negative prompt: \"{NEGATIVE_PREPROMPT}\").", default=True),
        scheduler: str = Input(description="The scheduler to use", default=SCHEDULER_NAMES[0], choices=SCHEDULER_NAMES),
        steps: int = Input(description="The steps when generating", default=35, ge=1, le=100),
        cfg_scale: float = Input(description="CFG Scale defines how much attention the model pays to the prompt when generating", default=7, ge=1, le=50),
        guidance_rescale: float = Input(description="The amount to rescale CFG generated noise to avoid generating overexposed images", default=0.7, ge=0, le=5),
        width: int = Input(description="The width of the image", default=1184, ge=1, le=4096),
        height: int = Input(description="The height of the image", default=864, ge=1, le=4096),
        strength: float = Input(description="How much noise to add (For image to image and inpainting only, larger value indicates more noise added to the input image)", default=0.7, ge=0, le=1),
        blur_factor: float = Input(description="The factor to blur the inpainting mask for smoother transition between masked and unmasked", default=5, ge=0),
        batch_size: int = Input(description="Number of images to generate (1-4)", default=1, ge=1, le=4),
        seed: int = Input(description="The seed used when generating, set to -1 for random seed", default=-1),
    ) -> list[Path]:
        if prompt == "__ignore__":
            return []
        if prepend_preprompt:
            prompt = POSITIVE_PREPROMPT + prompt
            negative_prompt = NEGATIVE_PREPROMPT + negative_prompt
        gen_kwargs = {
            "prompt": prompt, "negative_prompt": negative_prompt, "num_inference_steps": steps,
            "guidance_scale": cfg_scale, "guidance_rescale": guidance_rescale, "num_images_per_prompt": batch_size,
        }
        pipeline = self.pipelines.get_pipeline(model, vae, scheduler)
        pipeline.unload_lora_weights()
        if lora_url:
            utils.process_lora(lora_url, pipeline)
        if image:
            gen_kwargs["image"] = utils.scale_and_crop(image, width, height)
            gen_kwargs["strength"] = strength
            if mask:
                # inpainting
                mask_img = utils.scale_and_crop(mask, width, height)
                pipeline = StableDiffusionXLInpaintPipeline.from_pipe(pipeline)
                mask_img = pipeline.mask_processor.blur(mask_img, blur_factor)
                gen_kwargs["mask_image"] = mask_img
                gen_kwargs["width"] = width
                gen_kwargs["height"] = height
                print("Using inpainting mode.")
            else:
                # img2img
                pipeline = StableDiffusionXLImg2ImgPipeline.from_pipe(pipeline)
                print("Using image to image mode.")
        else:
            if mask:
                raise ValueError("You must upload a base image for inpainting mode.")
            # txt2img
            gen_kwargs["width"] = width
            gen_kwargs["height"] = height
            print("Using text to image mode.")
        if seed == -1:
            seed = random.randint(0, 2147483647)
        gen_kwargs["generator"] = torch.Generator(device="cuda").manual_seed(seed)
        print("Using seed:", seed)
        imgs = pipeline(**gen_kwargs).images

        image_paths = []
        for index, img in enumerate(imgs):
            img_file_path = f"tmp/{index}.png"
            img.save(img_file_path, compression=9)
            image_paths.append(Path(img_file_path))

        return image_paths

class SDXLMultiPipelineSwitchAutoDetect:

    def __init__(self, model_name_obj_dict, vaes_dir_path, vae_names, textual_inversion_paths):
        self.model_name_obj_dict = model_name_obj_dict
        self.model_pipeline_dict = {} # Key = Model's name(str), Value = StableDiffusionXLPipeline instance.
        self.vaes_dir_path = vaes_dir_path
        self.vae_obj_dict = {vae_name: None for vae_name in vae_names} # Key = VAE's name(str), Value = AutoencoderKL instance.
        self.textual_inversion_paths = textual_inversion_paths

        self._load_all_vaes() # Must load VAEs before models.
        self._load_all_models()

        self.on_cuda_model = model_names[0]
        on_cuda_pipeline = self.model_pipeline_dict[self.on_cuda_model]
        on_cuda_pipeline.to("cuda")
        on_cuda_pipeline.vae = self.vae_obj_dict[vae_names[0]]

    def get_pipeline(self, model_name, vae_name, scheduler_name):
        # __init__ function guarantees all models and VAEs to be loaded.
        pipeline = self.model_pipeline_dict.get(model_name)
        if pipeline is None:
            raise ValueError(f"Model \"{model_name}\" not found.")

        vae_name = model_name if vae_name == DEFAULT_VAE_NAME else vae_name
        vae = self.vae_obj_dict.get(vae_name)
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

    # Load all VAEs to GPU(CUDA).
    def _load_all_vaes(self):
        for vae_name in self.vae_obj_dict:
            self.vae_obj_dict[vae_name] = self._load_vae(vae_name)

    # Load a VAE to GPU(CUDA).
    def _load_vae(self, vae_name):
        vae = AutoencoderKL.from_pretrained(os.path.join(self.vaes_dir_path, vae_name), torch_dtype=TORCH_DTYPE)
        vae.enable_slicing()
        vae.enable_tiling()
        vae.to("cuda")
        return vae

    # Load all models to CPU.
    def _load_all_models(self):
        clip_l_list, clip_g_list, activation_token_list = utils.get_textual_inversions(self.textual_inversion_paths)
        for model_name, model_for_loading in self.model_name_obj_dict.items():
            self.model_pipeline_dict[model_name] = self._load_model(model_name, model_for_loading, clip_l_list, clip_g_list, activation_token_list)

    # Load a model to CPU.
    def _load_model(self, model_name, model_for_loading, clip_l_list, clip_g_list, activation_token_list):
        model_loading_kwargs = {"torch_dtype": TORCH_DTYPE, "add_watermarker": False}
        if model_for_loading.is_single_file:
            pipeline = StableDiffusionXLPipeline.from_single_file(model_for_loading.model_path, **model_loading_kwargs)
        else:
            pipeline = StableDiffusionXLPipeline.from_pretrained(model_for_loading.model_path, **model_loading_kwargs)
        utils.apply_textual_inversions_to_sdxl_pipeline(pipeline, clip_l_list, clip_g_list, activation_token_list)
        vae = pipeline.vae
        pipeline.vae = None
        vae.enable_slicing()
        vae.enable_tiling()
        vae.to("cuda")
        self.vae_obj_dict[model_name] = vae
        return pipeline
