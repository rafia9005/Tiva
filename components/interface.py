import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

def generate_image(prompt, pipe):
    """Menghasilkan gambar dari prompt yang diberikan."""
    image = pipe(prompt, num_inference_steps=100).images[0]

    plt.imshow(image)
    plt.axis('off') 
    plt.show()

def create_ui(pipe):
    prompt_text = widgets.Text(
        description='Prompt:',
        placeholder='Masukkan prompt di sini...'
    )

    generate_button = widgets.Button(description='Generate Image')

    def on_generate_button_clicked(b):
        prompt = prompt_text.value
        if prompt:
            generate_image(prompt, pipe)
        else:
            print("Tolong masukkan prompt yang valid.")

    generate_button.on_click(on_generate_button_clicked)

    display(prompt_text, generate_button)

