from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

# Load model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").tokenizer

# List of image URLs
image_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",  # Image 1
    "http://images.cocodataset.org/val2017/000000212226.jpg",  # Image 2
]

# Load images
images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]

# Define text prompts
text = ["dog", "Two cats", "A person and a cow"]

# Preprocess images and text
image_inputs = processor(images=images, return_tensors="pt")
print("Images_input:", image_inputs)
text_inputs = processor(text=text, return_tensors="pt", padding=True)
print("Text_inputs:", text_inputs)

token_id = torch.tensor([tokenizer.convert_tokens_to_ids(text[0])])
token_id = text_inputs['input_ids']
print("Token_Ids:", token_id)

# Get embeddings
with torch.no_grad():
    image_embeddings = model.get_image_features(**image_inputs)  # Shape: [num_images, 512]
    text_embeddings = model.get_text_features(**text_inputs)  # Shape: [num_texts, 512]
    token_embedding = model.text_model.embeddings.token_embedding(token_id)
# Print embeddings
print("Image Embedding Shape:", image_embeddings.shape)  # [num_images, 512]
print("Image Embeddings:\n", image_embeddings)

print("\nText Embedding Shape:", text_embeddings.shape)  # [num_texts, 512]
print("Text Embeddings:\n", text_embeddings)

print("\nToken Embedding Shape:", token_embedding.shape)  # [num_texts, 512]
print("Token Embeddings:\n", token_embedding)

image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
# Compute cosine similarity (dot product after normalization)
similarity_matrix = image_embeddings @ text_embeddings.T  # Shape: [num_images, num_texts]

# Print results
print("\nSimilarity Matrix (Image-to-Text):")
print(similarity_matrix)  # Values between 0 and 1 (higher = more similar)

'''
# Display images
fig, axes = plt.subplots(1, len(images), figsize=(10, 5))
for idx, img in enumerate(images):
    axes[idx].imshow(img)
    axes[idx].axis("off")
plt.show()
'''