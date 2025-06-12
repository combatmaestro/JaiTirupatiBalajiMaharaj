import torch
from PIL import Image
from models.e4e_encoder import E4EEncoder
from models.stylegan2_generator import StyleGAN2Generator
import numpy as np
import cv2

def average_latents(latents):
    return torch.mean(torch.stack(latents), dim=0, keepdim=True)  # Shape: [1, 512]

def get_blended_face(image_list, stylegan_path, e4e_path, use_cuda):
    device = 'cuda' if use_cuda else 'cpu'
    encoder = E4EEncoder(e4e_path, device=device)
    generator = StyleGAN2Generator(stylegan_path, device=device)

    # Convert all images to PIL.Image
    pil_images = []
    for img in image_list:
        if isinstance(img, np.ndarray):
            pil_images.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        elif isinstance(img, Image.Image):
            pil_images.append(img)
        else:
            raise ValueError("Unsupported image type in get_blended_face")

    # Encode images to latent vectors
    latents = [encoder.encode(pil_img) for pil_img in pil_images]
    avg_latent = average_latents(latents)  # [1, 512]

    # StyleGAN2 expects [1, 18, 512]
    repeated_latent = avg_latent.repeat(1, generator.n_latent, 1)  # [1, 18, 512]

    result_image = generator.synthesize(repeated_latent)

    # Convert result to OpenCV format (BGR)
    return cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
