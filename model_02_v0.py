from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch
import os
import gradio as gr

# Model ID and local directory
model_id = "SG161222/RealVisXL_V4.0_Lightning"

# Load the pipeline
try:
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir="D:/Huggingface Model")
    pipe = pipe.to("cuda")
    print("Pipeline loaded successfully")
except Exception as e:
    print(f"Error loading pipeline: {e}")

# Set scheduler to DPMSolverMultistepScheduler
try:
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    print("Scheduler set successfully")
except Exception as e:
    print(f"Error setting scheduler: {e}")

# Define the prompt and negative prompt
default_prompt = "powerful raw portrait photo of a man,dark scene,window light, 22 y.o. , natural eyes, natural skin, hard shadows, authentic high film grain, still 22mm photo"
default_negative_prompt = "(octane render, render, drawing, anime, bad photo, bad photography:1.3), (worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips, long neck), (bad anatomy, bad proportions:1.1), (deformed iris, deformed pupils), (deformed eyes, bad eyes), (deformed face, ugly face, bad face), (deformed hands, bad hands, fused fingers), morbid, mutilated, mutation, disfigured"

# Function to generate image
def generate_image(prompt, negative_prompt, num_inference_steps, guidance_scale, seed, height, width):
    generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None
    try:
        with torch.no_grad():
            output = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator, height=height, width=width, negative_prompt=negative_prompt)
            image = output.images[0]
            print("Image generated successfully")
            return image
    except Exception as e:
        print(f"Error during image generation: {e}")
        return None

# Function to save image with automatic renaming
def save_image(image, base_name, extension="jpg"):
    file_name = f"{base_name}.{extension}"
    counter = 1
    while os.path.exists(file_name):
        file_name = f"{base_name}_{counter}.{extension}"
        counter += 1
    image.save(file_name)
    print(f"Saved image as {file_name}")

# Define the Gradio interface
def gradio_interface(prompt=default_prompt, negative_prompt=default_negative_prompt, num_inference_steps=35, guidance_scale=5.0, seed=52454212, height=640, width=960):
    print(f"Generating image with prompt: {prompt}")
    print(f"Negative prompt: {negative_prompt}")
    print(f"Steps: {num_inference_steps}, Guidance Scale: {guidance_scale}, Seed: {seed}, Height: {height}, Width: {width}")
    image = generate_image(prompt, negative_prompt, num_inference_steps, guidance_scale, seed, height, width)
    if image:
        save_image(image, "img/model-02/model02-photo_")
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
