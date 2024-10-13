import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

def generate_image(prompt, pipe, num_steps, img_size):
    if img_size == "Square":
        height, width = 256, 256
    elif img_size == "Landscape":
        height, width = 256, 512
    elif img_size == "Portrait":
        height, width = 512, 256

    image = pipe(prompt, num_inference_steps=num_steps, height=height, width=width).images[0]

    plt.imshow(image)
    plt.axis('off') 
    plt.show()

def create_ui(pipe):
    prompt_text = widgets.Text(
        description='Prompt:',
        placeholder='Masukkan prompt di sini...'
    )

    steps_slider = widgets.IntSlider(
        value=100,
        min=10,
        max=200,
        step=10,
        description='Inference Steps:',
        continuous_update=False
    )

    size_dropdown = widgets.Dropdown(
        options=["Square", "Landscape", "Portrait"],
        value="Square",
        description='Image Size:',
    )

    generate_button = widgets.Button(description='Generate Image')

    def on_generate_button_clicked(b):
        prompt = prompt_text.value
        num_steps = steps_slider.value
        img_size = size_dropdown.value
        if prompt:
            generate_image(prompt, pipe, num_steps, img_size)
        else:
            print("Tolong masukkan prompt yang valid.")

    generate_button.on_click(on_generate_button_clicked)

    display(prompt_text, steps_slider, size_dropdown, generate_button)

