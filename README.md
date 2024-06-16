# articulate-ai (TBD)

# Installation
## Stable_Diffusion
### Cloning Project
Before cloning the Stable Diffusion repository, change to the parent directory of the PTSD project:
```sh
cd ..
```

Clone the Stable Diffusion Repository:
```sh
git clone https://github.com/VincentLi1216/stable-diffusion.git
```

Change directory into the `stable-diffusion` folder:
```
cd stable-diffusion
```

Run `./stable_diffusion/webui.sh`. This will download dependencies on the first run (approximately 10 minutes).   
Ensure to include `--api` at the end of the command, as the system will not operate without it:
```sh
sh webui.sh --api
```

### Downloading Models
#### Method1: Google Drive
1. Navigate to **[My Google Drive](https://drive.google.com/drive/u/0/folders/1sGHcNInrKNdsd3m_XNn9Q8Zveya1S7uS)**, and download `sd_xl_base_1.0.safetensors` and `lcm_lora_sdxl.safetensors`.
2. Move `sd_xl_base_1.0.safetensors` to `./models/Stable-diffusion`.
3. Move `lcm_lora_sdxl.safetensors` to `./models/Lora`
#### Method2: Hugging Face
1. Go to [Hugging Face URL1](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main) and download `sd_xl_base_1.0.safetensors`
2. Go to [Hugging Face URL2](https://huggingface.co/latent-consistency/lcm-lora-sdxl/blob/main/pytorch_lora_weights.safetensors) and download `pytorch_lora_weights.safetensors`
3. Rename `pytorch_lora_weights.safetensors` to `lcm_lora_sdxl.safetensors`
4. Move `sd_xl_base_1.0.safetensors` to `./models/Stable-diffusion`.
5. Move `lcm_lora_sdxl.safetensors` to `./models/Lora`

### Rerunning the system
Close the previous section:
```
ctrl+c
```

Run it once again.  
Ensure to include **--api** at the end of the command, as the system will not operate without it:
```sh
sh webui.sh --api
```

