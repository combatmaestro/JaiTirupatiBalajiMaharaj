import os
import cv2
import requests
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from cloudinary.uploader import upload as cloud_upload
from cloudinary import config as cloud_config
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from gfpgan import GFPGANer

# ---- Cloudinary Configuration ----
cloud_config(
    cloud_name="djpclyujw",
    api_key="915755835229494",
    api_secret="IBZLafEyhV80nlbww46Kp7u-izY",
    secure=True
)

# ---- Model Paths ----
MODEL_DIR = 'models'
INSWAPPER_PATH = 'inswapper_128.onnx'
GFPGAN_PATH = os.path.join(MODEL_DIR, 'GFPGANv1.4.pth')

# ---- Initialize FaceAnalysis ----
face_analyzer = FaceAnalysis(
    name='buffalo_l',
    root=MODEL_DIR,
    download=False,
    providers=['CPUExecutionProvider'],
    allowed_modules=["detection", "recognition", "landmark_2d_106", "landmark_3d_68"]
)
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# ---- Load Face Swap Model ----
swapper = get_model(INSWAPPER_PATH, providers=['CPUExecutionProvider'])

# ---- Load GFPGAN Enhancer ----
gfpgan = GFPGANer(
    model_path=GFPGAN_PATH,
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
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


# ---- Utility ----
def url_to_image(url: str) -> np.ndarray:
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        img_array = np.asarray(bytearray(resp.content), dtype="uint8")
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {url}")

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

            for variation in page.variations:
                target_img = url_to_image(variation.target_image_url)
                target_faces = face_analyzer.get(target_img)

                if not target_faces:
                    print(f"❌ No face in target image: {variation.target_image_url}")
                    continue

                for tgt_face in tgt_faces:
                    swapped = swapper.get(swapped, tgt_face, src_face, paste_back=True)
                # Enhance with GFPGAN
                _, _, enhanced = gfpgan.enhance(
                    swapped,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )

                temp_path = f"/tmp/page-{page.pageNumber}-{variation.variation}.jpg"
                cv2.imwrite(temp_path, enhanced)

                uploaded = cloud_upload(temp_path)
                page_result["variations"][variation.variation] = uploaded["secure_url"]

            output.append(page_result)

        return {"pages": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")
