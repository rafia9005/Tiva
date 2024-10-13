from components.interface import create_ui
from models.utils import load_stable_diffusion_model

def main():
    model_id = "CompVis/stable-diffusion-v1-4"
    
    pipe = load_stable_diffusion_model(model_id)

    create_ui(pipe)

if __name__ == "__main__":
    main()

