import os
import json
import utils # utils.py
import torch
import safetensors
from constants import * # constants.py

def _validate(url, strength=1, civitai_token=None):
    if url is None:
        raise KeyError("URL isn't provided!")
    if not isinstance(url, str):
        raise ValueError("URL isn't a string!")
    url = url.strip()
    if not url:
        raise ValueError("URL is empty!")

    if strength is None:
        strength = 1
    if not isinstance(strength, int) and not isinstance(strength, float):
        raise ValueError("Strength isn't a number!")
    strength = float(strength)

    if civitai_token is not None:
        if isinstance(civitai_token, str):
            civitai_token = civitai_token.strip()
            if not civitai_token:
                civitai_token = None
        else:
            raise ValueError("CivitAI token isn't a string!")

    return url, strength, civitai_token

def _parse_json(text):
    j = json.loads(text)
    if not isinstance(j, list):
        raise ValueError("The LoRA JSON isn't a list!")
    return [_validate(e.get("url"), e.get("strength"), e.get("civitai_token")) for e in j]

def _parse_plain(text):
    parts = text.split(",")
    ret = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        sub_parts = part.rsplit(":", 1)
        try:
            strength = float(sub_parts[1])
        except (ValueError, IndexError):
            sub_parts[0] = ":".join(sub_parts)
            strength = 1
        ret.append(_validate(sub_parts[0], strength, None))
    return ret

def parse(text):
    if not text:
        return []
    text = text.strip()
    if not text:
        return []
    try:
        return _parse_json(text)
    except json.JSONDecodeError:
        pass
    except Exception as e:
        raise e.__class__("The LoRA JSON is invalid! Error: " + str(e)) from e
    return _parse_plain(text)

class SDXLMultiLoRAHandler:

    def __init__(self):
        self.lora_dict = {} # Key: URL, Value: LoRA filename.
        self.lora_cache = {} # Key: LoRA filename, Value: (LoRA weight, Size in bytes).

    def limit_cache(self):
        overflowed_bytes = sum(weight_size_tuple[1] for weight_size_tuple in self.lora_cache.values()) - MAX_LORA_CACHE_BYTES
        for filename in list(self.lora_cache):
            if overflowed_bytes <= 0:
                break
            size = self.lora_cache.pop(filename)[1]
            overflowed_bytes -= size

    def get_state_dicts(self, loras):
        state_dicts = []
        for url, strength, civitai_token in loras:
            if url not in self.lora_dict:
                download_url, filename = utils.get_download_info(url, None if civitai_token is None else {"__Secure-civitai-token": civitai_token})
                _, ext = os.path.splitext(filename)
                if ext not in {".safetensors", ".bin"}:
                    raise RuntimeError(f"URL file extension \"{ext}\" isn't supported for LoRA!")
                lora_path = os.path.join(LORAS_DIR_PATH, filename)
                if not os.path.isfile(lora_path):
                    utils.download(download_url, lora_path)
                self.lora_dict[url] = filename
                self.lora_dict[download_url] = filename
            else:
                filename = self.lora_dict[url]
                _, ext = os.path.splitext(filename)
                lora_path = os.path.join(LORAS_DIR_PATH, filename)

            state_dict = self.lora_cache.get(filename)
            if state_dict is None:
                if ext == ".safetensors":
                    state_dict = safetensors.torch.load_file(lora_path)
                else:
                    state_dict = torch.load(lora_path)
                for name, weight in state_dict.items():
                    state_dict[name] = weight.to(TORCH_DTYPE)
                self.lora_cache[filename] = (state_dict, sum(tensor.numel() * 2 for tensor in state_dict.values()))
                self.limit_cache()
            else:
                state_dict = state_dict[0]

            state_dicts.append((filename.replace(".", "_"), strength, state_dict))
        return state_dicts

    def process(self, text, pipeline):
        loras = parse(text)
        if not loras:
            return
        state_dicts = self.get_state_dicts(loras)
        for state_dict in state_dicts:
            pipeline.load_lora_weights(state_dict[2], state_dict[0])
        lora_targets = []
        lora_strengths = []
        for state_dict in state_dicts:
            lora_targets.append(state_dict[0])
            lora_strengths.append(state_dict[1])
        pipeline.set_adapters(lora_targets, lora_strengths)
