import os
import torch
from torchvision import transforms
from PIL import Image

class E4EEncoder:
    def __init__(self, model_path='models/e4e_ffhq_encode.pt', device='cuda'):
        if not os.path.exists(model_path):
            import urllib.request
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print("📥 Downloading e4e model...")
            urllib.request.urlretrieve(
                'https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models/encoder4editing/e4e_ffhq_encode.pt',
                model_path
            )
            print("✅ e4e model downloaded.")

        self.device = device
        ckpt = torch.load(model_path, map_location=device)

        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            from models.psp import pSp
            opts_dict = ckpt['opts']
            opts_dict['checkpoint_path'] = model_path
            opts = type('obj', (object,), opts_dict)  # dict to object
            self.model = pSp(opts)
            self.model.load_state_dict(ckpt['state_dict'])
        else:
            self.model = ckpt

        self.model.to(device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def encode(self, image: Image.Image):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            result = self.model(img_tensor, input_code=False, return_latents=True)
            if isinstance(result, tuple):
                _, latent = result
            else:
                latent = result

        # Fix: Remove extra batch dim if accidentally present
        if latent.ndim == 4 and latent.shape[0] == 1:
            latent = latent.squeeze(0)  # From [1, 1, 18, 512] → [1, 18, 512]
        elif latent.ndim == 3 and latent.shape[1] == 1:
            latent = latent.squeeze(1)  # From [1, 1, 512] → [1, 512] (for W space)

        return latent


