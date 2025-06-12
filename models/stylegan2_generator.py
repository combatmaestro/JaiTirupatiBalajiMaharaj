import os
import torch

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

        self.device = device
        try:
            ckpt = torch.load(model_path, map_location=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load the model from {model_path}: {e}")

        if isinstance(ckpt, dict) and 'g_ema' in ckpt:
            self.g = ckpt['g_ema'].to(device)
        elif hasattr(ckpt, 'to'):  # sometimes torch.load returns a nn.Module directly
            self.g = ckpt.to(device)
        else:
            raise TypeError("Loaded model is not in expected format (dict with 'g_ema' or model object).")

        self.g.eval()

    def synthesize(self, latent):
        with torch.no_grad():
            img_tensor = self.g(latent, None)
        img = ((img_tensor.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1).squeeze().cpu().numpy()
        return img.astype('uint8')
