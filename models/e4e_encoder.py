import os
import torch
from torchvision import transforms
from PIL import Image

class E4EEncoder:
    def __init__(self, model_path='models/e4e_ffhq_encode.pt', device='cuda'):
        if not os.path.exists(model_path):
            import urllib.request
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print("\U0001F4E5 Downloading e4e model...")
            urllib.request.urlretrieve(
                'https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models/encoder4editing/e4e_ffhq_encode.pt',
                model_path
            )

        self.device = device
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def encode(self, image: Image.Image):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.model(img_tensor)
        return latent