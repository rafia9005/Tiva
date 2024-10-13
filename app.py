import torch
from diffusers import StableDiffusionPipeline

# load model and scheduler
model_id = "CompVis/stable-diffusion-v1-4"
# if you dont have vgs
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None).to("cpu")

# if use vga
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None).to("cuda")

# generate image with prompt
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

prompt = "Sebuah pemandangan gunung di bawah sinar matahari"
generated_image = generate_image(prompt)
