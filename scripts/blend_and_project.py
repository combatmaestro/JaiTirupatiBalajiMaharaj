from models.e4e_encoder import E4EEncoder
from models.stylegan2_generator import StyleGAN2Generator
import torch
import numpy as np
from PIL import Image


def average_latents(latents):
    return torch.mean(torch.stack(latents), dim=0, keepdim=True)


def blend_faces(image_list: list[Image.Image], device='cuda'):
    encoder = E4EEncoder(device=device)
    generator = StyleGAN2Generator(device=device)

    # Encode all faces
    latents = []
    for img in image_list:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        latents.append(encoder.encode(img))

    # Blend latents
    avg_latent = average_latents(latents)

    # Decode into an image
    blended_img = generator.synthesize(avg_latent)
    return blended_img