import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from gfpgan import GFPGANer

# ------------ Config ------------
SOURCE_IMAGE = 'source.png'
TARGET_IMAGE = 'page2.png'
OUTPUT_IMAGE = 'final_magic_swap.jpg'
ONNX_MODEL = 'inswapper_128.onnx'
GFPGAN_MODEL = 'experiments/pretrained_models/GFPGANv1.4.pth'
DEVICE = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
# --------------------------------

# Step 1: Enhance source face with GFPGAN
print("🔧 Enhancing source face...")
restorer = GFPGANer(
    model_path=GFPGAN_MODEL,
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None,
    device=DEVICE
)
src_img = cv2.imread(SOURCE_IMAGE)
_, _, enhanced_source = restorer.enhance(src_img, has_aligned=False, only_center_face=True, paste_back=True)

# Step 2: Initialize InsightFace
print("📸 Initializing InsightFace...")
face_app = FaceAnalysis(
    name='buffalo_l',
    root='.',
    providers=['CPUExecutionProvider'],
    allowed_modules=["landmark_3d_68", "landmark_2d_106", "detection", "recognition"]
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Step 3: Detect faces
target_img = cv2.imread(TARGET_IMAGE)
src_faces = face_app.get(enhanced_source)
tgt_faces = face_app.get(target_img)

if not src_faces or not tgt_faces:
    print("❌ Face detection failed.")
    exit()

src_face = src_faces[0]
tgt_face = tgt_faces[0]

# Step 4: Perform face swap
print("🔁 Swapping face...")
swap_model = get_model(ONNX_MODEL, providers=['CPUExecutionProvider'])
swapped_img = swap_model.get(target_img.copy(), tgt_face, src_face, paste_back=True)

# Step 5: Build full-face polygon mask
print("🎭 Building face mask...")
mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
landmarks = tgt_face.landmark_3d_68.astype(int)

face_outline = list(range(0, 17)) + list(reversed(range(17, 22))) + list(range(22, 27))
face_contour = np.array([landmarks[i][:2] for i in face_outline], dtype=np.int32)
cv2.fillConvexPoly(mask, face_contour, 255)
mask = cv2.GaussianBlur(mask, (65, 65), 30)

# Step 6A: Remove cartoon eyes before blending
print("🧼 Cleaning cartoon eyes...")
clean_target = target_img.copy()
eye_mask = np.zeros_like(mask)

left_eye = [36, 37, 38, 39, 40, 41]
right_eye = [42, 43, 44, 45, 46, 47]
eye_points = np.array([landmarks[i][:2] for i in left_eye + right_eye], dtype=np.int32)
cv2.fillConvexPoly(eye_mask, eye_points, 255)
eye_mask = cv2.GaussianBlur(eye_mask, (25, 25), 12)
clean_target = cv2.inpaint(clean_target, eye_mask, 3, cv2.INPAINT_TELEA)

# Step 6B: Blend swapped face over cleaned target
print("🪄 Blending...")
output = swapped_img.astype(np.float32) * (mask[..., None] / 255.0) + \
         clean_target.astype(np.float32) * (1 - (mask[..., None] / 255.0))
output = np.clip(output, 0, 255).astype(np.uint8)

# Step 6C: Sharpen swapped area
print("✨ Sharpening face...")
sharpened = cv2.GaussianBlur(output, (0, 0), 3)
output = cv2.addWeighted(output, 1.5, sharpened, -0.5, 0)

# Step 7: Optional cartoon-style smoothing
output = cv2.bilateralFilter(output, d=9, sigmaColor=75, sigmaSpace=75)

# Step 8: Save
cv2.imwrite(OUTPUT_IMAGE, output)
print(f"✅ Done! Output saved as: {OUTPUT_IMAGE}")
