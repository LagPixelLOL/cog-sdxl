import os
import glob

# Returns the base filenames of the models.
def find_models(models_dir):
    return [os.path.basename(file) for file in glob.glob(f"{models_dir}/**/*.safetensors", recursive=True) + glob.glob(f"{models_dir}/**/*.ckpt", recursive=True)]

# Returns the folder names of the VAEs.
def find_vaes(vaes_dir):
    vae_names = []
    for folder in os.listdir(vaes_dir):
        folder_path = os.path.join(vaes_dir, folder)
        if os.path.isdir(folder_path):
            safetensors_file = os.path.join(folder_path, 'diffusion_pytorch_model.safetensors')
            bin_file = os.path.join(folder_path, 'diffusion_pytorch_model.bin')
            config_file = os.path.join(folder_path, 'config.json')
            if (os.path.isfile(safetensors_file) or os.path.isfile(bin_file)) and os.path.isfile(config_file):
                vae_names.append(folder)
    return vae_names

# Returns the relative paths of the textual inversions.
def find_textual_inversions(textual_inversions_dir):
    return [file for file in glob.glob(f"{textual_inversions_dir}/**/*.safetensors", recursive=True) + glob.glob(f"{textual_inversions_dir}/**/*.bin", recursive=True)]
