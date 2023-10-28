# modules/sdxl_pipeline.py
import diffusers
import torch
from diffusers import StableDiffusionXLPipeline
from constants.schedulers import SCHEDULERS

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
