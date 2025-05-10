from enum import Enum
from diffusers import (
    UniPCMultistepScheduler,
    HeunDiscreteScheduler,
    DDIMScheduler,
    KDPM2AncestralDiscreteScheduler,
    DPMSolverSDEScheduler,
    DDPMScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    KDPM2DiscreteScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
)

class SDXLCompatibleSchedulers(Enum):
    EulerA = ("Euler a", EulerAncestralDiscreteScheduler, {})
    UniPC = ("UniPC", UniPCMultistepScheduler, {})
    Heun = ("Heun", HeunDiscreteScheduler, {})
    DDIM = ("DDIM", DDIMScheduler, {})
    DPM2A = ("DPM2 a", KDPM2AncestralDiscreteScheduler, {})
    DPM2AKarras = ("DPM2 a Karras", KDPM2AncestralDiscreteScheduler, {"use_karras_sigmas": True})
    DPMSDE = ("DPM SDE", DPMSolverSDEScheduler, {})
    DDPM = ("DDPM", DDPMScheduler, {})
    DPMPlusPlusSDE = ("DPM++ SDE", DPMSolverSinglestepScheduler, {})
    DPMPlusPlusSDEKarras = ("DPM++ SDE Karras", DPMSolverSinglestepScheduler, {"use_karras_sigmas": True})
    LMS = ("LMS", LMSDiscreteScheduler, {})
    LMSKarras = ("LMS Karras", LMSDiscreteScheduler, {"use_karras_sigmas": True})
    Euler = ("Euler", EulerDiscreteScheduler, {})
    PNDM = ("PNDM", PNDMScheduler, {})
    DPM2 = ("DPM2", KDPM2DiscreteScheduler, {})
    DPM2Karras = ("DPM2 Karras", KDPM2DiscreteScheduler, {"use_karras_sigmas": True})
    DEIS = ("DEIS", DEISMultistepScheduler, {})
    DPMPlusPlus2M = ("DPM++ 2M", DPMSolverMultistepScheduler, {})
    DPMPlusPlus2MKarras = ("DPM++ 2M Karras", DPMSolverMultistepScheduler, {"use_karras_sigmas": True})
    DPMPlusPlus2MSDE = ("DPM++ 2M SDE", DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++"})
    DPMPlusPlus2MSDEKarras = ("DPM++ 2M SDE Karras", DPMSolverMultistepScheduler, {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"})

    def __init__(self, string_name, scheduler_class, init_args):
        self.string_name = string_name
        self.scheduler_class = scheduler_class
        self.init_args = init_args

    @classmethod
    def create_instance(cls, name, v_pred=False):
        for scheduler in cls:
            if scheduler.string_name == name:
                scheduler_kwargs = {"pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0", "subfolder": "scheduler", **scheduler.init_args}
                if v_pred: scheduler_kwargs.update({"prediction_type": "v_prediction", "rescale_betas_zero_snr": True})
                return scheduler.scheduler_class.from_pretrained(**scheduler_kwargs)
        raise ValueError(f"Scheduler with name \"{name}\" does not exist in SDXLCompatibleSchedulers.")

    @classmethod
    def get_names(cls):
        return [scheduler.string_name for scheduler in cls]
