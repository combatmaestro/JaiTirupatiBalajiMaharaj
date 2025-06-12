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
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)  # Shape: [1, 3, 256, 256]
        print(f"📥 Encoding image... Input tensor shape: {img_tensor.shape}")

        with torch.no_grad():
            result = self.model(img_tensor, input_code=False, return_latents=True)

        if isinstance(result, tuple):
            _, latent = result
        else:
            latent = result

        print(f"🔍 Raw latent shape from model: {latent.shape}")

        # 🔥 Fix the latent shape here
        while latent.ndim > 3:
            latent = latent.squeeze(0)
            print(f"✂️ Squeezing latent, new shape: {latent.shape}")

        if latent.ndim == 2:
            latent = latent.unsqueeze(0)
            print(f"📦 Unsqueezed to final shape: {latent.shape}")

        print(f"✅ Final latent shape: {latent.shape}")
        return latent  # Final shape: [1, 18, 512]
