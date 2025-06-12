import os
import cv2
import requests
import numpy as np
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from cloudinary.uploader import upload as cloud_upload
from cloudinary import config as cloud_config
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer
from concurrent.futures import ThreadPoolExecutor
import onnxruntime as ort
from PIL import Image
from io import BytesIO
import traceback
import torch
from torch import nn
from models.e4e_encoder import E4EEncoder
from models.stylegan2_generator import StyleGAN2Generator

# ---- Cloudinary Configuration ----
cloud_config(
    cloud_name="djpclyujw",
    api_key="915755835229494",
    api_secret="IBZLafEyhV80nlbww46Kp7u-izY",
    secure=True
)

# ---- Model Paths ----
MODEL_DIR = 'models'
GFPGAN_PATH = os.path.join(MODEL_DIR, 'GFPGANv1.4.pth')
STYLEGAN_PATH = os.path.join(MODEL_DIR, 'stylegan2-ffhq-config-f.pt')
E4E_ENCODER_PATH = os.path.join(MODEL_DIR, 'e4e_ffhq_encode.pt')

# ---- Download if Missing ----
def download_if_missing():
    files = {
        GFPGAN_PATH: 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
        STYLEGAN_PATH: 'https://huggingface.co/Awesimo/jojogan/resolve/main/stylegan2-ffhq-config-f.pt',
        E4E_ENCODER_PATH: 'https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models/encoder4editing/e4e_ffhq_encode.pt'
    }
    for path, url in files.items():
        if not os.path.exists(path):
            print(f"\U0001F4E5 Downloading {os.path.basename(path)}...")
            import urllib.request
            urllib.request.urlretrieve(url, path)
            print(f"✅ {os.path.basename(path)} downloaded.")

# ---- Setup ----
download_if_missing()

available_providers = ort.get_available_providers()
use_cuda = 'CUDAExecutionProvider' in available_providers
print(f"\U0001F680 Available Providers: {available_providers}")
print(f"✅ Using: {'cuda' if use_cuda else 'cpu'}")

# ---- Initialize Models ----
gfpgan = GFPGANer(
    model_path=GFPGAN_PATH,
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None,
    device='cuda' if use_cuda else 'cpu'
)

# ---- FastAPI Models ----
class VariationInput(BaseModel):
    variation: str
    target_image_url: str

class PageInput(BaseModel):
    pageNumber: int
    text: str
    variations: List[VariationInput]

class SwapRequest(BaseModel):
    face_image_urls: List[str]
    pages: List[PageInput]

# ---- App ----
app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)

# ---- Utility: Load URL image safely ----
def url_to_image(url: str, max_size: int = 768) -> np.ndarray:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")
        w, h = pil_image.size
        if w < 128 or h < 128:
            raise ValueError("Image too small")
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            pil_image = pil_image.resize((int(w * scale), int(h * scale)))
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from {url} | {e}")

# ---- Identity-Preserving Interpolation ----
def optimize_identity(generator, latent, target_emb, face_analyzer, steps=2, lr=0.01):
    latent = latent.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        synth = generator.synthesize(latent)
        synth_np = cv2.cvtColor(np.array(synth.detach().cpu().squeeze().permute(1,2,0).numpy()*255, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        faces = face_analyzer.get(synth_np)
        if not faces:
            continue
        emb = torch.tensor(faces[0].embedding, dtype=torch.float32).to(latent.device)
        target_emb_tensor = torch.tensor(target_emb, dtype=torch.float32).to(latent.device)
        id_loss = 1 - torch.nn.functional.cosine_similarity(emb.unsqueeze(0), target_emb_tensor.unsqueeze(0)).mean()
        id_loss.backward()
        optimizer.step()
    return latent.detach()

# ---- New Blending Logic: Interpolated + Optimized Head ----
def interpolate_and_generate_head(source_img: np.ndarray, target_img: np.ndarray, stylegan_path, e4e_path, alpha=0.6):
    device = 'cuda' if use_cuda else 'cpu'
    encoder = E4EEncoder(e4e_path, device=device)
    generator = StyleGAN2Generator(stylegan_path, device=device)
    face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'] if use_cuda else ['CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0 if use_cuda else -1)

    source_pil = Image.fromarray(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)).resize((256, 256))
    target_pil = Image.fromarray(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)).resize((256, 256))

    source_latent = encoder.encode(source_pil)
    target_latent = encoder.encode(target_pil)

    blended_latent = (1 - alpha) * target_latent + alpha * source_latent
    if blended_latent.ndim == 2:
        blended_latent = blended_latent.unsqueeze(1).repeat(1, 18, 1)

    ref_embedding = face_analyzer.get(source_img)[0].embedding
    optimized_latent = optimize_identity(generator, blended_latent, ref_embedding, face_analyzer)

    result_image = generator.synthesize(optimized_latent)
    return cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)

# ---- Face blend + enhance ----
def process_variation(source_img: np.ndarray, page_number, variation: VariationInput):
    try:
        target_img = url_to_image(variation.target_image_url)
        head = interpolate_and_generate_head(source_img, target_img, STYLEGAN_PATH, E4E_ENCODER_PATH)

        h, w, _ = target_img.shape
        face_resized = cv2.resize(head, (w // 2, h // 2))
        x_offset = w // 4
        y_offset = h // 4
        blended_img = target_img.copy()
        blended_img[y_offset:y_offset+face_resized.shape[0], x_offset:x_offset+face_resized.shape[1]] = face_resized

        _, _, enhanced = gfpgan.enhance(blended_img, has_aligned=False, only_center_face=True, paste_back=True)

        with tempfile.NamedTemporaryFile(suffix=f"_{page_number}_{variation.variation}.jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, enhanced)
            uploaded = cloud_upload(tmp.name)
            return variation.variation, uploaded["secure_url"]

    except Exception as e:
        print(f"❌ Error in {variation.variation}: {e}")
        return variation.variation, None

# ---- Endpoint ----
@app.post("/swap-batch")
async def swap_batch(data: SwapRequest):
    try:
        src_faces = [cv2.cvtColor(url_to_image(url), cv2.COLOR_BGR2RGB) for url in data.face_image_urls]
        source_img = src_faces[0]

        output = []
        for page in data.pages:
            page_result = {"pageNumber": page.pageNumber, "text": page.text, "variations": {}}
            futures = [executor.submit(process_variation, source_img, page.pageNumber, variation)
                       for variation in page.variations]
            for future in futures:
                var_name, var_url = future.result()
                if var_url:
                    page_result["variations"][var_name] = var_url
            output.append(page_result)

        return {"pages": output}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")
