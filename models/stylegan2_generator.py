import os
import torch
import pickle
from PIL import Image
from models.stylegan2.model import Generator  # Make sure this exists and is correct


class StyleGAN2Generator:
    def __init__(self, model_path='models/stylegan2-ffhq-config-f.pt', device='cuda'):
        if not os.path.exists(model_path):
            import urllib.request
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print("📥 Downloading StyleGAN2 model...")
            urllib.request.urlretrieve(
                'https://huggingface.co/Awesimo/jojogan/resolve/main/stylegan2-ffhq-config-f.pt',
                model_path
            )

        ckpt = torch.load(model_path, map_location=device)

        if 'g_ema' not in ckpt:
            raise ValueError("Expected 'g_ema' in checkpoint but not found.")

        self.device = device
        self.g = Generator(1024, 512, 8)  # 1024x1024 resolution, 512-dim latent, 8 layers
        self.g.load_state_dict(ckpt['g_ema'])
        self.g.to(device).eval()
        self.n_latent = self.g.n_latent

    def synthesize(self, latent):
        with torch.no_grad():
            if latent.ndim == 2:
                latent = latent.unsqueeze(1).repeat(1, self.n_latent, 1)  # [B, 512] -> [B, 18, 512]
            img_tensor, _ = self.g([latent], input_is_latent=True, randomize_noise=False)

        # Convert to image
        img = ((img_tensor.clamp(-1, 1) + 1) / 2 * 255)
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy().astype('uint8')
        return Image.fromarray(img)
