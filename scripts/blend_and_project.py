import torch
from PIL import Image
from models.e4e_encoder import E4EEncoder
from models.stylegan2_generator import StyleGAN2Generator
import numpy as np
import cv2

def average_latents(latents):
    print("🧪 Latents count:", len(latents))
    for idx, lt in enumerate(latents):
        print(f"  🔹 Latent[{idx}] shape: {lt.shape}")
    return torch.mean(torch.stack(latents), dim=0, keepdim=True)  # Expected: [1, 18, 512]

def get_blended_face(image_list, stylegan_path, e4e_path, use_cuda):
    device = 'cuda' if use_cuda else 'cpu'
    encoder = E4EEncoder(e4e_path, device=device)
    generator = StyleGAN2Generator(stylegan_path, device=device)

    print("📷 Converting input images...")
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
    latents = [encoder.encode(pil_img) for pil_img in pil_images]

    avg_latent = average_latents(latents)
    print("📏 Average latent shape BEFORE normalization:", avg_latent.shape)

    # 🔍 Debug the shape here
    if avg_latent.ndim == 2 and avg_latent.shape[1] == 512:
        print("🔄 Expanding [1, 512] → [1, 1, 512] → repeat to [1, 18, 512]")
        avg_latent = avg_latent.unsqueeze(1)
        repeated_latent = avg_latent.repeat(1, generator.n_latent, 1)
    elif avg_latent.ndim == 3:
        if avg_latent.shape[1] == 1 and avg_latent.shape[2] == 512:
            print("🔁 Repeating [1, 1, 512] → [1, 18, 512]")
            repeated_latent = avg_latent.repeat(1, generator.n_latent, 1)
        elif avg_latent.shape[1] == 18:
            print("✅ Latent already shape [1, 18, 512]")
            repeated_latent = avg_latent
        else:
            print("❌ Unsupported shape (ndim=3):", avg_latent.shape)
            raise ValueError(f"Unsupported latent shape: {avg_latent.shape}")
    else:
        print("❌ Unexpected latent shape:", avg_latent.shape)
        raise ValueError(f"Unexpected latent shape: {avg_latent.shape}")

    print("🖼️ Synthesizing final face from latent...")
    result_image = generator.synthesize(repeated_latent)

    return cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
