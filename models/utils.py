import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

def load_stable_diffusion_model(model_id="CompVis/stable-diffusion-v1-4"):
    """Load the Stable Diffusion model and pipeline."""
    
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    
    # Ganti [cuda] menjadi [cpu] jika tidak memiliki GPU
    pipe = pipe.to("cuda")
    
    return pipe

