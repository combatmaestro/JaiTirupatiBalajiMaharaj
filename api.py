import os
import cv2
import requests
import numpy as np
import insightface
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from cloudinary.uploader import upload as cloud_upload
from cloudinary import config as cloud_config

# Auto-setup models if not found
def setup_models():
    os.makedirs("models", exist_ok=True)

    # Download buffalo_l if missing
    if not os.path.exists("models/buffalo_l"):
        os.system("git clone https://github.com/deepinsight/insightface.git tempface && mv tempface/model_zoo/buffalo_l models/ && rm -rf tempface")

    # Download inswapper if missing
    if not os.path.exists("models/inswapper_128.onnx"):
        import urllib.request
        print("⬇️ Downloading inswapper_128.onnx...")
        url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
        urllib.request.urlretrieve(url, "models/inswapper_128.onnx")

setup_models()

# Cloudinary config
cloud_config(
    cloud_name="djpclyujw",
    api_key="915755835229494",
    api_secret="IBZLafEyhV80nlbww46Kp7u-izY",
    secure=True
)

# Init face swap
# Detect GPU availability
try:
    import onnxruntime
    available_providers = onnxruntime.get_available_providers()
    providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]
    print(f"🔧 Using provider: {providers[0]}")
except:
    providers = ["CPUExecutionProvider"]

face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', root='models', providers=providers)
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model("models/inswapper_128.onnx", providers=providers)

# Models
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

# App
app = FastAPI()

def url_to_image(url):
    resp = requests.get(url)
    img_array = np.asarray(bytearray(resp.content), dtype="uint8")
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def blend_face(src_face, tgt_face, target_img):
    swapped = swapper.get(target_img.copy(), tgt_face, src_face, paste_back=True)

    landmarks = tgt_face.landmark_3d_68[:, :2].astype(np.int32)
    mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(landmarks), 255)
    mask = cv2.GaussianBlur(mask, (15, 15), 5)

    alpha = mask[..., None] / 255.0
    output = (swapped.astype(np.float32) * alpha +
              target_img.astype(np.float32) * (1.0 - alpha))
    output = np.clip(output, 0, 255).astype(np.uint8)

    sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(output, -1, sharpen)

@app.post("/swap-batch")
async def swap_batch(data: SwapRequest):
    try:
        src_img = url_to_image(data.face_image_url)
        src_faces = face_analyzer.get(src_img)
        if not src_faces:
            raise HTTPException(status_code=400, detail="No face found in face_image_url")

        src_face = src_faces[0]
        final_output = []

        for page in data.pages:
            page_out = {"pageNumber": page.pageNumber, "text": page.text, "variations": {}}
            for variation in page.variations:
                tgt_img = url_to_image(variation.target_image_url)
                tgt_faces = face_analyzer.get(tgt_img)
                if not tgt_faces:
                    continue
                result = blend_face(src_face, tgt_faces[0], tgt_img)
                temp_path = f"/tmp/page-{page.pageNumber}-{variation.variation}.jpg"
                cv2.imwrite(temp_path, result)
                cloud_result = cloud_upload(temp_path)
                page_out["variations"][variation.variation] = cloud_result["secure_url"]
            final_output.append(page_out)

        return {"pages": final_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
