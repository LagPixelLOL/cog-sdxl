# Cog-SDXL
Inference SDXL with Cog including multiple models in 1 instance support.

There are some settings in `constants.py` which you can change to modify the behaviors of this script.

This repo supports 3 modes of model loading:
- Remote Single File: Put `model_name.remote` files into the `models` folder, the container will download the weights every time during startup, this will speed cold boot by 3x on Replicate because if the weights are bundled into the image, when Replicate pull the Docker image, docker need to decompress it, this is actually way slower compared to high speed download from HuggingFace or other platforms. This repo uses [PGet](https://github.com/replicate/pget), also made by Replicate, to download the weights in multiple chunks at the same time, which can achieve a speed of 10~20 Gbps on Replicate when downloading files from HuggingFace, a typical (b)float16 is 7GB, which can be downloaded in around 5 seconds, decompressing the weights would take way longer. During my experiment, an image with 3 models takes only 1 minute to cold start using remote single file compared to using local multi file. If you are using this repo on Replicate, this option is highly recommended.
- Local Single File: Put single file weights `model_name.safetensors` into the `models` folder, the weights will be bundled into the image, this will make huge images but it won't download the models each time it starts up.
- Local Multi File: Put multi file weights (HuggingFace Diffusers format) `model_name` folders into the `models` folder, this is similar to local single file.

Note:
- `model_name` is a placeholder, you can change it to whatever you seem fit.
- Remote Multi File isn't supported at this time but maybe in the future will.

If you want to convert single file `model_name.safetensors` into multi file HuggingFace Diffusers format, you can use the `extract_model.py` script like:
```bash
python extract_model.py model_name.safetensors
```
This will convert the model into HuggingFace Diffusers format and delete the old single file weight, if you don't want the script to delete the old single file weight, you can pass the `-n` argument to the script.

Use below command to push:
```bash
sudo cog push r8.im/user/repo --separate-weights
```
