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
from insightface.model_zoo import get_model
from gfpgan import GFPGANer
from concurrent.futures import ThreadPoolExecutor
import onnxruntime as ort
from PIL import Image
from io import BytesIO
import traceback
from scripts.blend_and_project import get_blended_face

# ---- Cloudinary Configuration ----
cloud_config(
    cloud_name="djpclyujw",
    api_key="915755835229494",
    api_secret="IBZLafEyhV80nlbww46Kp7u-izY",
    secure=True
)

# ---- Model Paths ----
MODEL_DIR = 'models'
INSWAPPER_PATH = os.path.join(MODEL_DIR, 'inswapper_128.onnx')
GFPGAN_PATH = os.path.join(MODEL_DIR, 'GFPGANv1.4.pth')
STYLEGAN_PATH = os.path.join(MODEL_DIR, 'stylegan2-ffhq-config-f.pt')
E4E_ENCODER_PATH = os.path.join(MODEL_DIR, 'e4e_ffhq_encode.pt')

# ---- Download if Missing ----
def download_if_missing():
    files = {
        INSWAPPER_PATH: 'https://huggingface.co/combatmaestro/inswapper_128.onxx/resolve/main/inswapper_128.onnx',
        GFPGAN_PATH: 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
        STYLEGAN_PATH: 'https://huggingface.co/Awesimo/jojogan/resolve/main/stylegan2-ffhq-config-f.pt',
        E4E_ENCODER_PATH: 'https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models/encoder4editing/e4e_ffhq_encode.pt'
    }
    for path, url in files.items():
        if not os.path.exists(path):
            print(f"📥 Downloading {os.path.basename(path)}...")
            import urllib.request
            urllib.request.urlretrieve(url, path)
            print(f"✅ {os.path.basename(path)} downloaded.")

# ---- Setup ----
download_if_missing()

available_providers = ort.get_available_providers()
use_cuda = 'CUDAExecutionProvider' in available_providers
preferred_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
print(f"🚀 Available Providers: {available_providers}")
print(f"✅ Using: {preferred_providers[0]}")

# ---- Initialize Models ----
try:
    face_analyzer = FaceAnalysis(
        name='buffalo_l',
        root=MODEL_DIR,
        download=False,
        allowed_modules=["detection", "recognition", "landmark_2d_106", "landmark_3d_68"],
        providers=["CUDAExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    )
    face_analyzer.prepare(ctx_id=0 if use_cuda else -1, det_size=(640, 640))
except Exception as e:
    print(f"❌ InsightFace GPU load failed, fallback to CPU: {e}")
    face_analyzer = FaceAnalysis(
        name='buffalo_l',
        root=MODEL_DIR,
        download=False,
        allowed_modules=["detection", "recognition", "landmark_2d_106", "landmark_3d_68"],
        providers=["CPUExecutionProvider"]
    )
    face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))

swapper = get_model(INSWAPPER_PATH, providers=preferred_providers)

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

# ---- Face swap + enhance ----
def process_variation(blended_face: np.ndarray, page_number, variation: VariationInput):
    try:
        target_img = url_to_image(variation.target_image_url)
        target_faces = face_analyzer.get(target_img)
        if not target_faces:
            print(f"❌ No face in: {variation.target_image_url}")
            return variation.variation, None

        # Convert the blended_face to a Face object
        blended_faces = face_analyzer.get(blended_face)
        if not blended_faces:
            print(f"❌ No face found in blended face image for variation {variation.variation}")
            return variation.variation, None

        blended_face_obj = blended_faces[0]

        swapped = swapper.get(target_img, target_faces[0], blended_face_obj, paste_back=True)

        if swapped is None or not isinstance(swapped, np.ndarray):
            print(f"⚠️ Swapped image invalid for {variation.variation}")
            return variation.variation, None

        try:
            _, _, enhanced = gfpgan.enhance(
                swapped,
                has_aligned=False,
                only_center_face=True,
                paste_back=True
            )
        except Exception as e:
            print(f"⚠️ GFPGAN failed for {variation.variation}: {e}")
            return variation.variation, None

        if enhanced is None or not isinstance(enhanced, np.ndarray):
            print(f"❌ Enhanced image invalid for {variation.variation}")
            return variation.variation, None

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
        blended_face = get_blended_face(src_faces, STYLEGAN_PATH, E4E_ENCODER_PATH, use_cuda)

        output = []
        for page in data.pages:
            page_result = {"pageNumber": page.pageNumber, "text": page.text, "variations": {}}
            futures = [executor.submit(process_variation, blended_face, page.pageNumber, variation)
                       for variation in page.variations]
            for future in futures:
                var_name, var_url = future.result()
                if var_url:
                    page_result["variations"][var_name] = var_url
            output.append(page_result)

        return {"pages": output}

    except Exception as e:
        print("❌ Exception in /swap-batch:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")
