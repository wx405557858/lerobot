import time
import requests
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# model_name = "openai/clip-vit-large-patch14"
device = "cuda"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

print(f"output embedding dim: {model.text_projection.out_features}, {model.visual_projection.out_features}")

# Example for text embedding
text_inputs = processor(text=["Pick up the orange ring and move to reset position", "Place the orange ring above character A and release", "Place the orange ring above character B and release"], return_tensors="pt", padding=True).to(device)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
for i in range(10):
    start_time = time.time()
    text_embeddings = model.get_text_features(**text_inputs)
    end_time = time.time()
    print(f"Iteration {i}: {end_time - start_time} seconds")

# inputs = processor(images=image, return_tensors="pt")
image_inputs = processor(images=image, return_tensors="pt").to(device)
image_embeds = model.get_image_features(**image_inputs)

# print("Text Embeddings:", text_embeddings)
print("Text Embeddings:", text_embeddings.size())
print("distances:", ((text_embeddings[:]-text_embeddings[1]).norm(2, dim=-1)))

print("Image Embeddings:", image_embeds.size())

# for names, param in model.named_parameters():
#     print(names, param.requires_grad, param.device)