import torch
from diffusers import StableDiffusion3Pipeline
import gradio as gr
from huggingface_hub import login

login()

# Load the model
repo = "stabilityai/stable-diffusion-3-medium-diffusers"
pipe = StableDiffusion3Pipeline.from_pretrained(repo, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Load LoRA weights
pipe.load_lora_weights("nerijs/pixel-art-medium-128-v0.1", weight_name="pixel-art-medium-128-v0.1.safetensors", cache_dir="D:/Huggingface Model")

# Function to generate image
def generate_image(prompt, negative_prompt):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt
    ).images[0]
    return image

# Gradio interface
gr_interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
        gr.Textbox(label="Negative Prompt", placeholder="Enter your negative prompt here...")
    ],
    outputs=gr.Image(label="Generated Image")
)

# Launch the interface
gr_interface.launch()
