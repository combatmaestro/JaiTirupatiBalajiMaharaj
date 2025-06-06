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
from concurrent.futures import ThreadPoolExecutor
from gfpgan import GFPGANer

# ------------------ Setup ------------------
# Auto-setup models if not found
def setup_models():
    os.makedirs("models", exist_ok=True)

    if not os.path.exists("models/buffalo_l"):
        os.system("git clone https://github.com/deepinsight/insightface.git tempface && mv tempface/model_zoo/buffalo_l models/ && rm -rf tempface")

    if not os.path.exists("models/inswapper_128.onnx"):
        import urllib.request
        print("⬇️ Downloading inswapper_128.onnx...")
        url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
        urllib.request.urlretrieve(url, "models/inswapper_128.onnx")

    if not os.path.exists("models/GFPGANv1.4.pth"):
        os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P models")

setup_models()

# Cloudinary config
cloud_config(
    cloud_name="djpclyujw",
    api_key="915755835229494",
    api_secret="IBZLafEyhV80nlbww46Kp7u-izY",
    secure=True
)

# Detect GPU or fallback
try:
    import onnxruntime
    available_providers = onnxruntime.get_available_providers()
    providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]
    print(f"🔧 Using provider: {providers[0]}")
except:
    providers = ["CPUExecutionProvider"]

# Load InsightFace
face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', root='models', providers=providers)
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model("models/inswapper_128.onnx", providers=providers)

# Load GFPGAN
gfpgan = GFPGANer(
    model_path="models/GFPGANv1.4.pth",
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

# Thread executor
executor = ThreadPoolExecutor(max_workers=5)

# ------------------ Models ------------------
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

# ------------------ App Setup ------------------
app = FastAPI()

def url_to_image(url):
    try:
        resp = requests.get(url, timeout=15)
        img_array = np.asarray(bytearray(resp.content), dtype="uint8")
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"❌ Error downloading image: {url}")
        raise e

# ------------------ Swap Endpoint ------------------
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
            futures = []

            def process_variation(page_num, variation_obj):
                try:
                    tgt_img = url_to_image(variation_obj.target_image_url)
                    tgt_faces = face_analyzer.get(tgt_img)
                    if not tgt_faces:
                        return (variation_obj.variation, None)

                    result = swapper.get(tgt_img.copy(), tgt_faces[0], src_face, paste_back=True)

                    # Enhance with GFPGAN only
                    _, _, restored_img = gfpgan.enhance(result, has_aligned=False, only_center_face=False, paste_back=True)

                    temp_path = f"/tmp/page-{page_num}-{variation_obj.variation}.jpg"
                    cv2.imwrite(temp_path, restored_img)
                    cloud_result = cloud_upload(temp_path)
                    os.remove(temp_path)
                    return (variation_obj.variation, cloud_result["secure_url"])
                except Exception as e:
                    print(f"❌ Error in variation {variation_obj.variation}: {e}")
                    return (variation_obj.variation, None)

            for v in page.variations:
                futures.append(executor.submit(process_variation, page.pageNumber, v))

            for future in futures:
                var_name, url = future.result()
                if url:
                    page_out["variations"][var_name] = url

            final_output.append(page_out)

        return {"pages": final_output}

    except Exception as e:
        print(f"❌ Exception in /swap-batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
