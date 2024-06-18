from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch
import os
import gradio as gr
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from PIL import Image
import numpy as np

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

# Load Real-ESRGAN model for upscaling
def load_realesrgan():
    print("Loading Real-ESRGAN model...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_path = 'Real-ESRGAN/4x_NMKD-Superscale-SP_178000_G.pth'  # Change this to the correct path

    # Load the state dict
    state_dict = torch.load(model_path, map_location='cuda')
    
    # Inspect the state dictionary
    if 'params' in state_dict:
        state_dict = state_dict['params']
    
    # Load the state dict with a strict check
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Model loaded with strict=True")
    except RuntimeError as e:
        print(f"RuntimeError with strict=True: {e}")
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded with strict=False")

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device='cuda'
    )
    print("Real-ESRGAN model loaded successfully")
    return upsampler

# Function to upscale image
def upscale_image(image, upsampler):
    print("Upscaling image...")
    image = image.convert("RGB")
    image = np.array(image)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().div(255.0).cuda()
    upscaled_image, _ = upsampler.enhance(image)
    upscaled_image = upscaled_image.squeeze().permute(1, 2, 0).cpu().numpy()
    upscaled_image = (upscaled_image * 255).astype(np.uint8)
    print("Upscaling completed.")
    return Image.fromarray(upscaled_image)

# Function to generate image
def generate_image(prompt, negative_prompt, num_inference_steps, guidance_scale, seed, height, width, hires_fix, hires_steps, denoising_strength, upscale_by):
    generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None
    try:
        with torch.no_grad():
            output = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator, height=height, width=width, negative_prompt=negative_prompt)
            image = output.images[0]
            print("Image generated successfully")
            
            if hires_fix:
                print("Applying Hires Fix...")
                hires_height = int(height * upscale_by)
                hires_width = int(width * upscale_by)
                output = pipe(prompt, num_inference_steps=hires_steps, guidance_scale=guidance_scale, generator=generator, height=hires_height, width=hires_width, negative_prompt=negative_prompt, denoising_strength=denoising_strength)
                image = output.images[0]
                print("Hires Fix applied successfully")
                
            return image
    except Exception as e:
        print(f"Error during image generation: {e}")
        return None

# Function to save image with automatic renaming
def save_image(image, base_name, extension="jpeg"):
    file_name = f"{base_name}.{extension}"
    counter = 1
    while os.path.exists(file_name):
        file_name = f"{base_name}_{counter}.{extension}"
        counter += 1
    image.save(file_name)
    print(f"Saved image as {file_name}")

# Define the Gradio interface
def gradio_interface(prompt=default_prompt, negative_prompt=default_negative_prompt, num_inference_steps=35, guidance_scale=5.0, seed=52454212, height=640, width=960, hires_fix=False, hires_steps=10, denoising_strength=0.2, upscale_by=1.2, upscale=False):
    print(f"Generating image with prompt: {prompt}")
    print(f"Negative prompt: {negative_prompt}")
    print(f"Steps: {num_inference_steps}, Guidance Scale: {guidance_scale}, Seed: {seed}, Height: {height}, Width: {width}")
    print(f"Hires Fix: {hires_fix}, Hires Steps: {hires_steps}, Denoising Strength: {denoising_strength}, Upscale By: {upscale_by}, Upscale: {upscale}")
    image = generate_image(prompt, negative_prompt, num_inference_steps, guidance_scale, seed, height, width, hires_fix, hires_steps, denoising_strength, upscale_by)
    if image:
        if upscale:
            print("Loading upscaler...")
            upsampler = load_realesrgan()
            print("Upscaler loaded. Starting upscaling process...")
            image = upscale_image(image, upsampler)
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
        gr.Slider(minimum=256, maximum=1024, step=64, value=832, label="Width"),
        gr.Checkbox(label="Apply Hires Fix", value=False),
        gr.Slider(minimum=1, maximum=50, step=1, value=10, label="Hires Steps"),
        gr.Slider(minimum=0.1, maximum=0.3, step=0.01, value=0.2, label="Denoising Strength"),
        gr.Slider(minimum=1.1, maximum=1.5, step=0.1, value=1.2, label="Upscale By"),
        gr.Checkbox(label="Upscale Image", value=False)
    ],
    outputs=gr.Image(type="pil"),
    title="Stable Diffusion Image Generator",
    description="Generate images using Stable Diffusion with custom prompts and settings.",
)

# Launch the interface
interface.launch()
