import gradio as gr
from PIL import Image
import torch
import numpy as np
import os
from main import Generator

def load_model(model_path, device):
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    G_AB.load_state_dict(checkpoint['G_AB'])
    G_BA.load_state_dict(checkpoint['G_BA'])
    return G_AB, G_BA

def pil_to_tensor(image):
    # Resize and convert PIL image to normalized PyTorch tensor
    image = image.resize((256, 256)).convert("RGB")
    img_np = np.array(image).astype(np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5  # Normalize to [-1, 1]
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC to CHW
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # Add batch dim
    return img_tensor

def tensor_to_pil(tensor):
    # Convert tensor (BCHW) to PIL image
    tensor = tensor.squeeze(0).cpu()
    tensor = tensor * 0.5 + 0.5  # Denormalize from [-1, 1] to [0, 1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)

def process_image(input_image, direction, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    G_AB, G_BA = load_model(model_path, device)

    # Process input image
    img_tensor = pil_to_tensor(input_image).to(device)

    with torch.no_grad():
        if direction == "Photo to Sketch":
            output = G_AB(img_tensor)
        else:
            output = G_BA(img_tensor)

    # Convert output tensor to PIL Image
    return tensor_to_pil(output)

def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# Photo-Sketch Converter using CycleGAN")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                model_path = gr.Textbox(label="Model Checkpoint Path", value="C:\\Users\\aalia\\Downloads\\i210605_AliAshraf\\Frontend\\checkpoint_1.pth")
                direction = gr.Radio(["Photo to Sketch", "Sketch to Photo"], label="Conversion Direction", value="Photo to Sketch")
                submit_btn = gr.Button("Convert")

            with gr.Column():
                output_image = gr.Image(type="pil", label="Output Image")

        submit_btn.click(
            fn=process_image,
            inputs=[input_image, direction, model_path],
            outputs=output_image
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
