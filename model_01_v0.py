from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, KDPM2DiscreteScheduler
import torch
import os
import gradio as gr

# Model ID
model_id = "sam749/ICBINP-New-Year"

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir="D:/Huggingface Model")
pipe = pipe.to("cuda")

# Placeholder for DPM++ 2M Karras sampler
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)

# Define the prompt and negative prompt
default_prompt = "powerful raw portrait photo of a man,dark scene,window light, 22 y.o. , natural eyes, natural skin, hard shadows, authentic high film grain, still 22mm photo"
default_negative_prompt = "(octane render, render, drawing, anime, bad photo, bad photography:1.3), (worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips, long neck), (bad anatomy, bad proportions:1.1), (deformed iris, deformed pupils), (deformed eyes, bad eyes), (deformed face, ugly face, bad face), (deformed hands, bad hands, fused fingers), morbid, mutilated, mutation, disfigured"

# Function to generate image
def generate_image(prompt, negative_prompt, num_inference_steps, guidance_scale, seed, height, width):
    generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator, height=height, width=width, negative_prompt=negative_prompt).images[0]
    return image

# Function to save image with automatic renaming
def save_image(image, base_name, extension="png"):
    file_name = f"{base_name}.{extension}"
    counter = 1
    while os.path.exists(file_name):
        file_name = f"{base_name}_{counter}.{extension}"
        counter += 1
    image.save(file_name)
    print(f"Saved image as {file_name}")

# Define the Gradio interface
def gradio_interface(prompt=default_prompt, negative_prompt=default_negative_prompt, num_inference_steps=35, guidance_scale=5.0, seed=52454212, height=640, width=960):
    image = generate_image(prompt, negative_prompt, num_inference_steps, guidance_scale, seed, height, width)
    save_image(image, "img/model-01/model01-photo_")
    return image

# Create Gradio interface
interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your prompt here...", value=default_prompt, label="Prompt"),
        gr.Textbox(lines=2, placeholder="Enter your negative prompt here...", value=default_negative_prompt, label="Negative Prompt"),
        gr.Slider(minimum=1, maximum=50, step=1, value=20, label="Number of Inference Steps"),
        gr.Slider(minimum=1, maximum=20.0, step=0.1, value=4.0, label="Guidance Scale"),
        gr.Number(value=125512231, label="Seed (for reproducibility)"),
        gr.Slider(minimum=256, maximum=1024, step=64, value=896, label="Height"),
        gr.Slider(minimum=256, maximum=1024, step=64, value=832, label="Width")
    ],
    outputs=gr.Image(type="pil"),
    title="Stable Diffusion Image Generator",
    description="Generate images using Stable Diffusion with custom prompts and settings.",
)

# Launch the interface
interface.launch()
