import matplotlib.pyplot as plt
from models.utils import load_stable_diffusion_model

def main():
    model_id = "CompVis/stable-diffusion-v1-4"
    
    # Memuat model menggunakan fungsi dari utils.py
    pipe = load_stable_diffusion_model(model_id)

    prompt = "a photo of an astronaut riding a horse on mars"

    image = pipe(prompt, num_inference_steps=100).images[0]


    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
