import os
import utils # utils.py
import diffusers
from constants import * # constants.py
from dataclasses import dataclass

LOCAL_EXTS = {".safetensors", ".ckpt"}
REMOTE_EXT = ".remote"

@dataclass
class ModelForLoading:
    model_name: str
    model_uri: str
    is_single_file: bool
    is_remote: bool

    def load(self, **model_loading_kwargs):
        if self.is_remote:
            if self.is_single_file:
                if os.path.isdir(MODELS_REMOTE_CACHE_PATH):
                    model_path = None
                    for path in os.listdir(MODELS_REMOTE_CACHE_PATH):
                        if self.model_name == os.path.splitext(path)[0]:
                            model_path = os.path.join(MODELS_REMOTE_CACHE_PATH, path)
                            break
                    if model_path is None:
                        model_path = self.download()
                else:
                    model_path = self.download()
                return diffusers.StableDiffusionXLPipeline.from_single_file(model_path, **model_loading_kwargs)
            else:
                raise NotImplementedError("Loading remote multi file weights (HuggingFace Diffusers format) isn't implemented!")
        else:
            if self.is_single_file:
                return diffusers.StableDiffusionXLPipeline.from_single_file(model_for_loading.model_path, **model_loading_kwargs)
            else:
                return diffusers.StableDiffusionXLPipeline.from_pretrained(model_for_loading.model_path, **model_loading_kwargs)

    def download(self):
        download_url, filename = utils.get_download_info(self.model_uri)
        _, ext = os.path.splitext(filename)
        if ext not in LOCAL_EXTS:
            raise RuntimeError(f"URL file extension \"{ext}\" isn't supported for model!")
        model_path = os.path.join(MODELS_REMOTE_CACHE_PATH, self.model_name + ext)
        utils.download(download_url, model_path)
        return model_path

    @classmethod
    def find_models(cls, base_dir: str):
        models = []
        for path in os.listdir(base_dir):
            path = os.path.join(base_dir, path)
            name, ext = os.path.splitext(path)
            name = os.path.basename(name)
            if os.path.isfile(path):
                if ext in LOCAL_EXTS:
                    models.append(cls(name, path, True, False))
                elif ext == REMOTE_EXT:
                    with open(path, "r", encoding="utf8") as file:
                        remote_url = file.read().strip()
                    models.append(cls(name, remote_url, True, True))
            elif os.path.isdir(path):
                paths_in_dir = os.listdir(path)
                if len(paths_in_dir) == 1:
                    inner_path = os.path.join(path, paths_in_dir[0])
                    inner_name, inner_ext = os.path.splitext(inner_path)
                    inner_name = os.path.basename(inner_name)
                    if os.path.isfile(inner_path):
                        if inner_ext in LOCAL_EXTS:
                            models.append(cls(inner_name, inner_path, True, False))
                        elif inner_ext == REMOTE_EXT:
                            with open(inner_path, "r", encoding="utf8") as file:
                                remote_url = file.read().strip()
                            models.append(cls(name, remote_url, True, True))
                elif "model_index.json" in paths_in_dir:
                    models.append(cls(name, path, False, False))
        models.sort(key=lambda x: x.model_name)
        return models

    @classmethod
    def to_dict(cls, models: list):
        return {model.model_name: model for model in sorted(models, key=lambda x: x.model_name)}
