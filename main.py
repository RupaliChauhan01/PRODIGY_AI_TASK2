from diffusers import StableDiffusionPipeline
import torch

# Replace with your Hugging Face token
YOUR_TOKEN = "hf_your_token_here"

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    use_auth_token=YOUR_TOKEN,
    torch_dtype=torch.float16
).to("cuda")

# Prompt
prompt = "A futuristic city with flying cars, digital art"

# Generate image
image = pipe(prompt).images[0]

# Save image
image.save("generated_image.png")
print("Image saved as generated_image.png")