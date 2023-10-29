from enum import Enum
from diffusers import (
    DPMSolverMultistepScheduler,
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
)

class SDXLCompatibleSchedulers(Enum):
    DPMPlusPlus2MSDEKarras = ("DPM++ 2M SDE Karras", DPMSolverMultistepScheduler, {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"})
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
    EulerA = ("Euler a", EulerAncestralDiscreteScheduler, {})
    Euler = ("Euler", EulerDiscreteScheduler, {})
    PNDM = ("PNDM", PNDMScheduler, {})
    DPM2 = ("DPM2", KDPM2DiscreteScheduler, {})
    DPM2Karras = ("DPM2 Karras", KDPM2DiscreteScheduler, {"use_karras_sigmas": True})
    DEIS = ("DEIS", DEISMultistepScheduler, {})
    DPMPlusPlus2M = ("DPM++ 2M", DPMSolverMultistepScheduler, {})
    DPMPlusPlus2MKarras = ("DPM++ 2M Karras", DPMSolverMultistepScheduler, {"use_karras_sigmas": True})
    DPMPlusPlus2MSDE = ("DPM++ 2M SDE", DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++"})

    def __init__(self, string_name, scheduler_class, init_args):
        self.string_name = string_name
        self.scheduler_class = scheduler_class
        self.init_args = init_args

    @classmethod
    def create_instance(cls, name):
        for scheduler in cls:
            if scheduler.string_name == name:
                return scheduler.scheduler_class.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", **scheduler.init_args)
        raise ValueError(f"Scheduler with name \"{name}\" does not exist in SDXLCompatibleSchedulers.")

    @classmethod
    def get_names(cls):
        return [scheduler.string_name for scheduler in cls]
