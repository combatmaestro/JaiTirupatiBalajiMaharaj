import os
import torch
import pickle

class StyleGAN2Generator:
    def __init__(self, model_path='models/stylegan2-ffhq-config-f.pt', device='cuda'):
        if not os.path.exists(model_path):
            import urllib.request
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print("\U0001F4E5 Downloading StyleGAN2 model...")
            urllib.request.urlretrieve(
                'https://huggingface.co/Awesimo/jojogan/resolve/main/stylegan2-ffhq-config-f.pt',
                model_path
            )

        with open(model_path, 'rb') as f:
            self.g = pickle.load(f)['G_ema'].to(device)
        self.g.eval()
        self.device = device

    def synthesize(self, latent):
        with torch.no_grad():
            img_tensor = self.g(latent, None)
        img = ((img_tensor.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1).squeeze().cpu().numpy()
        return img.astype('uint8')