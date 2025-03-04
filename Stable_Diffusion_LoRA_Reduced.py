import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer
import pandas as pd
from PIL import Image
from tqdm import tqdm
import json
from pathlib import Path
from torchvision import transforms
import random
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 1. Dataset Preparation
class ArtCapDataset(Dataset):
    def __init__(self, json_path, image_dir, tokenizer, size=512):
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.size = size
        
        # Load JSON annotations
        with open(json_path) as f:
            self.annotations = json.load(f)
        
        # Create list of (image_path, captions) pairs
        self.samples = [
            (self.image_dir / img_name, captions)
            for img_name, captions in self.annotations.items()
        ]
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, captions = self.samples[idx]
        
        # Load image and apply transforms
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Randomly select one caption from the 5 available
        caption = random.choice(captions)
        
        # Tokenize caption
        tokenized = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return image, tokenized.input_ids.squeeze(0)

class LoRA(nn.Module):
    def __init__(self, r, input_dim, output_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, r, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(r, output_dim, bias=False)
    
    def forward(self, x):
        return self.linear2(self.dropout(self.linear1(x)))

def freeze_all_params(model):
    """Freeze all parameters in the model"""
    for param in model.parameters():
        param.requires_grad = False

class LoRAWrapper(nn.Module):
    def __init__(self, original_layer, lora_layer):
        super().__init__()
        self.original_layer = original_layer
        self.lora = lora_layer
        
        # Freeze original weights
        for param in original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.original_layer(x) + self.lora(x)

def inject_lora(unet, r=8, dropout=0.1):
    """Inject LoRA layers while keeping original model frozen"""
    freeze_all_params(unet)
    target_layers = {
        "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out[0]",
        "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out[0]"
    }

    lora_layers = nn.ModuleList()
    
    for name, module in unet.named_modules():
        if any(x in name for x in target_layers) and isinstance(module, nn.Linear):
            # Create LoRA adapter
            lora = LoRA(
                r=r,
                input_dim=module.in_features,
                output_dim=module.out_features,
                dropout=dropout
            ).to(device)
            
            # Create wrapper and replace layer
            wrapper = LoRAWrapper(module, lora).to(device)
            
            # Get parent module
            parent = unet
            parts = name.split('.')
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            # Replace original layer with wrapped version
            setattr(parent, parts[-1], wrapper)
            lora_layers.append(lora)

    return unet, lora_layers

# 2. Training Setup
# Load model components
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
pipe = pipe.to(device)

# Inject LoRA (use previous implementation)
pipe.unet, lora_layers = inject_lora(pipe.unet, r=8)
# Load dataset
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
dataset = ArtCapDataset(
    json_path="artcap_dataset/ArtCap.json",
    image_dir="artcap_dataset/images",
    tokenizer=tokenizer
)
dataset = Subset(dataset, list(range(10)))
# Test dataset
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

# Optimizer (only LoRA params)
optimizer = torch.optim.AdamW(lora_layers.parameters(), lr=1e-4)
noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    total_loss = 0
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for images, input_ids in progress:
        optimizer.zero_grad()
        
        # Convert images to latent space
        latents = pipe.vae.encode(images.to(device, dtype=torch.float32)).latent_dist.sample()
        latents = latents * 0.18215
        
        # Sample noise
        noise = torch.randn_like(latents)
        bs = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=device).long()
        # Add noise
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        # Get text embeddings
        text_embeddings = pipe.text_encoder(input_ids.to(device))[0]
        # Predict noise
        noise_pred = pipe.unet(noisy_latents, timesteps, text_embeddings).sample
        
        # Loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        progress.set_postfix(loss=loss.item())
    train_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")
# Save LoRA weights
torch.save(lora_layers.state_dict(), "artcap_lora_reduced.pth")