import torch
from torch import nn
from diffusers import StableDiffusionPipeline

class LoRA(nn.Module):
    def __init__(self, r, input_dim, output_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, r, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(r, output_dim, bias = False)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x  = self.linear2(x)
        return x 
    
class LoRAWrapper(nn.Module):
    """Wraps an existing linear layer with LoRA adapters"""
    def __init__(self, original_layer, r=8, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        
        # Freeze original weights
        for param in original_layer.parameters():
            param.requires_grad = False
            
        # Create LoRA adapter
        self.lora = LoRA(
            r=r,
            input_dim=original_layer.in_features,
            output_dim=original_layer.out_features,
            dropout=dropout
        )
        
    def forward(self, x):
        return self.original_layer(x) + self.lora(x)
    
def inject_lora(unet, r=8, dropout=0.1):
    """Replace target linear layers with LoRA-wrapped versions"""
    # Target layers: attention query/key/value/output projections
    target_layers = {
        "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out[0]",
        "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out[0]"
    }

    for name, module in unet.named_modules():
        # Find all attention projection layers
        if any(x in name for x in target_layers) and isinstance(module, nn.Linear):
            # Get parent module and attribute name
            parent = unet
            comps = name.split('.')
            attr_name = comps[-1]
            
            # Traverse to parent module
            for c in comps[:-1]:
                parent = getattr(parent, c)
                
            # Replace with wrapped layer
            wrapped = LoRAWrapper(module, r=r, dropout=dropout)
            setattr(parent, attr_name, wrapped)
    
    return unet

# Load base model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Inject LoRA into UNet
pipe.unet = inject_lora(pipe.unet, r=8)

# Verify only LoRA params are trainable
trainable_params = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")  # Should be ~0.5M for r=8