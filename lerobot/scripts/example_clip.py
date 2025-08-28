import time
from transformers import CLIPProcessor, CLIPModel

# model_name = "openai/clip-vit-large-patch14"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Example for text embedding
text_inputs = processor(text=["place the ring on the left"], return_tensors="pt", padding=True)
for i in range(10):
    start_time = time.time()
    text_embeddings = model.get_text_features(**text_inputs)
    end_time = time.time()
    print(f"Iteration {i}: {end_time - start_time} seconds")

# print("Text Embeddings:", text_embeddings)
print("Text Embeddings:", text_embeddings.size())