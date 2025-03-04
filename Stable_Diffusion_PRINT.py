from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt
import random
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion pipeline
model_id = "runwayml/stable-diffusion-v1-5"
if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16  # Use half-precision for faster inference
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id)

pipe = pipe.to(device)

# Optimize memory usage (optional)
pipe.enable_attention_slicing()

# Input text prompt
prompt1 = "cow with female farmer out in the field"
#prompt2 = "man jogging"
# Generate image
image1 = pipe(prompt1, num_inference_steps=50).images[0]  # Adjust steps as needed
#image2 = pipe(prompt2, num_inference_steps=50).images[0]  # Adjust steps as needed
#img = Image.open(image)

# Display the image
plt.imshow(image1)
plt.axis('off')  # Hide axes
plt.show()
print("Image generated successfully!")