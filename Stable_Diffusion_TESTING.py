from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrain(model_id, use_auth_token=True)
lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="IMAGE_GENERATION")
pipe.unet = get_peft_model(pipe.unet, lora_config)