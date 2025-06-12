import os
import torch
import pickle
from models.stylegan2.model import Generator  # <- Make sure this path exists

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

        self.device = device
        self.g = Generator(1024, 512, 8)  # FFHQ: 1024 resolution, 512 latent, 8 mapping layers
        self.g.load_state_dict(ckpt['g_ema'])
        self.g.to(device).eval()

    def synthesize(self, latent):
        with torch.no_grad():
            img_tensor, _ = self.g([latent], input_is_latent=True, randomize_noise=False)
        img = ((img_tensor.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1).squeeze().cpu().numpy()
        return img.astype('uint8')
