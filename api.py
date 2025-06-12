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

# ---- Download if Missing ----
if not os.path.exists(INSWAPPER_PATH):
    print("📥 Downloading inswapper_128.onnx...")
    import urllib.request
    urllib.request.urlretrieve(
        'https://huggingface.co/combatmaestro/inswapper_128.onxx/resolve/main/inswapper_128.onnx',
        INSWAPPER_PATH
    )
    print("✅ inswapper_128.onnx downloaded.")

if not os.path.exists(GFPGAN_PATH):
    print("📥 Downloading GFPGANv1.4.pth...")
    import urllib.request
    urllib.request.urlretrieve(
        'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
        GFPGAN_PATH
    )
    print("✅ GFPGANv1.4.pth downloaded.")

# ---- Determine Execution Providers ----
available_providers = ort.get_available_providers()
preferred_providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
print(f"🚀 Using ONNX Runtime provider: {preferred_providers[0]}")

# ---- Initialize Models ----
face_analyzer = FaceAnalysis(
    name='buffalo_l',
    root=MODEL_DIR,
    download=False,
    providers=[preferred_providers[0]],
    allowed_modules=["detection", "recognition", "landmark_2d_106", "landmark_3d_68"]
)
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

swapper = get_model(INSWAPPER_PATH, providers=[preferred_providers[0]])

gfpgan = GFPGANer(
    model_path=GFPGAN_PATH,
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None,
    device='cuda' if 'CUDAExecutionProvider' in [preferred_providers[0]] else 'cpu'
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
    face_image_url: str
    pages: List[PageInput]

# ---- App ----
app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)

# ---- Utility: URL to image with resizing + validation ----
def url_to_image(url: str, max_size: int = 768) -> np.ndarray:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Load with PIL
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")
        w, h = pil_image.size

        # Reject very small images
        if w < 128 or h < 128:
            raise ValueError("Image too small (<128x128)")

        # Resize if needed
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            pil_image = pil_image.resize((int(w * scale), int(h * scale)))

        # Convert to OpenCV BGR
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from {url} | {e}")

# ---- Face swap + enhancement per variation ----
def process_variation(src_face, page_number, variation: VariationInput):
    try:
        target_img = url_to_image(variation.target_image_url)
        target_faces = face_analyzer.get(target_img)
        if not target_faces:
            print(f"❌ No face in target image: {variation.target_image_url}")
            return variation.variation, None

        swapped = swapper.get(target_img, target_faces[0], src_face, paste_back=True)

        try:
            _, _, enhanced = gfpgan.enhance(
                swapped,
                has_aligned=False,
                only_center_face=True,
                paste_back=True
            )
        except Exception as e:
            print(f"⚠️ GFPGAN failed on variation {variation.variation}: {e}")
            enhanced = swapped  # fallback to face-swapped image

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, enhanced)
            uploaded = cloud_upload(tmp.name)
            return variation.variation, uploaded["secure_url"]

    except Exception as e:
        print(f"❌ Error processing {variation.variation}: {e}")
        return variation.variation, None

# ---- Endpoint ----
@app.post("/swap-batch")
async def swap_batch(data: SwapRequest):
    try:
        source_img = url_to_image(data.face_image_url)
        source_faces = face_analyzer.get(source_img)
        if not source_faces:
            raise HTTPException(status_code=400, detail="❌ No face found in face_image_url")
        src_face = source_faces[0]

        output = []

        for page in data.pages:
            page_result = {
                "pageNumber": page.pageNumber,
                "text": page.text,
                "variations": {}
            }

            futures = [
                executor.submit(process_variation, src_face, page.pageNumber, variation)
                for variation in page.variations
            ]

            for future in futures:
                var_name, var_url = future.result()
                if var_url:
                    page_result["variations"][var_name] = var_url

            output.append(page_result)

        return {"pages": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")
