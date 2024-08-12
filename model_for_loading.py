import os
from dataclasses import dataclass

SAFETENSORS_EXT = ".safetensors"
UNSAFE_EXT = ".ckpt"

@dataclass
class ModelForLoading:
    model_name: str
    model_path: str
    is_single_file: bool

    @classmethod
    def find_models(cls, base_dir: str):
        models = []
        for path in os.listdir(base_dir):
            path = os.path.join(base_dir, path)
            if os.path.isfile(path):
                if path.endswith(SAFETENSORS_EXT) or path.endswith(UNSAFE_EXT):
                    models.append(cls.from_path(path, True))
            elif os.path.isdir(path):
                paths_in_dir = os.listdir(path)
                if len(paths_in_dir) == 1:
                    model_path = os.path.join(path, paths_in_dir[0])
                    if os.path.isfile(model_path) and (model_path.endswith(SAFETENSORS_EXT) or model_path.endswith(UNSAFE_EXT)):
                        models.append(cls.from_path(model_path, True))
                elif "model_index.json" in paths_in_dir:
                    models.append(cls.from_path(path, False))
        models.sort(key=lambda x: x.model_name)
        return models

    @classmethod
    def from_path(cls, model_path: str, is_single_file: bool):
        return cls(os.path.splitext(os.path.basename(model_path))[0], model_path, is_single_file)

    @classmethod
    def to_dict(cls, models: list):
        return {model.model_name: model for model in sorted(models, key=lambda x: x.model_name)}
