from diffusers import StableDiffusionPipeline
import torch
from torch import nn

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
            )
            
            # Create wrapper and replace layer
            wrapper = LoRAWrapper(module, lora)
            
            # Get parent module
            parent = unet
            parts = name.split('.')
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            # Replace original layer with wrapped version
            setattr(parent, parts[-1], wrapper)
            lora_layers.append(lora)

    return unet, lora_layers

# Initialize pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Inject LoRA and get trainable parameters
pipe.unet, lora_layers = inject_lora(pipe.unet, r=8)

# Verify freezing
total_params = sum(p.numel() for p in pipe.unet.parameters())
trainable_params = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
lora_params = sum(p.numel() for p in lora_layers.parameters())

print(f"Total parameters: {total_params:,}")        # ~859,520,964
print(f"Trainable parameters: {trainable_params:,}")  # Should match lora_params
print(f"LoRA parameters: {lora_params:,}")           # Should be ~0.5M for r=8