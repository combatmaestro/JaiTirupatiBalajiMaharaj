import torch
from PIL import Image
from models.e4e_encoder import E4EEncoder
from models.stylegan2_generator import StyleGAN2Generator
import numpy as np
import cv2

def average_latents(latents):
    # latents: List of [1, 18, 512]
    latent_tensor = torch.cat(latents, dim=0)  # Shape: [N, 18, 512]
    avg = torch.mean(latent_tensor, dim=0, keepdim=True)  # Shape: [1, 18, 512]
    print(f"📏 Average latent shape: {avg.shape}")
    return avg  # Final shape: [1, 18, 512]

def get_blended_face(image_list, stylegan_path, e4e_path, use_cuda):
    device = 'cuda' if use_cuda else 'cpu'
    encoder = E4EEncoder(e4e_path, device=device)
    generator = StyleGAN2Generator(stylegan_path, device=device)

    print("📷 Converting input images to PIL...")
    pil_images = []
    for img in image_list:
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pil_images.append(pil_img)
        elif isinstance(img, Image.Image):
            pil_images.append(img)
        else:
            raise ValueError("Unsupported image type in get_blended_face")

    print("🎯 Encoding images into latents...")
    latents = []
    for i, pil_img in enumerate(pil_images):
        print(f"📥 Encoding image {i + 1}/{len(pil_images)}...")
        latent = encoder.encode(pil_img)
        print(f"🔍 Encoded latent shape: {latent.shape}")
        latents.append(latent)

    avg_latent = average_latents(latents)

    # Normalize shape to [1, 18, 512] if needed
    if avg_latent.ndim == 2 and avg_latent.shape[1] == 512:
        avg_latent = avg_latent.unsqueeze(1)
        repeated_latent = avg_latent.repeat(1, generator.n_latent, 1)
        print("🔄 Normalized from [1, 512] → [1, 18, 512]")
    elif avg_latent.ndim == 3:
        if avg_latent.shape[1] == 1 and avg_latent.shape[2] == 512:
            repeated_latent = avg_latent.repeat(1, generator.n_latent, 1)
            print("🔁 Normalized from [1, 1, 512] → [1, 18, 512]")
        elif avg_latent.shape[1] == 18:
            repeated_latent = avg_latent
            print("✅ Latent already shape [1, 18, 512]")
        else:
            raise ValueError(f"❌ Unsupported latent shape: {avg_latent.shape}")
    else:
        raise ValueError(f"❌ Unexpected latent shape: {avg_latent.shape}")

    print("🖼️ Synthesizing final blended face...")
    result_image = generator.synthesize(repeated_latent)
    return cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
