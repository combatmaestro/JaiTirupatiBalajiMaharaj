# pip install insightface onnxruntime matplotlib opencv-python

import cv2
import insightface
import numpy as np
from matplotlib import pyplot as plt

# Setup
providers = ["CPUExecutionProvider"]
target_frame = cv2.imread('page1-girl-variation1.png')  # Use highest-resolution target
src_frame = cv2.imread('source.png')

# Initialize FaceAnalysis
face_app = insightface.app.FaceAnalysis(
    name='buffalo_l',
    root='.',
    providers=providers,
    allowed_modules=["landmark_3d_68", "landmark_2d_106", "detection", "recognition"]
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Detect faces
src_faces = face_app.get(src_frame)
tgt_faces = face_app.get(target_frame)
if not src_faces or not tgt_faces:
    print("❌ Face detection failed.")
    exit()

# Load face swapper
model = insightface.model_zoo.get_model('./inswapper_128.onnx', providers=providers)

# Swap face with full paste
swapped_img = model.get(
    img=target_frame.copy(),
    target_face=tgt_faces[0],
    source_face=src_faces[0],
    paste_back=True
)

# Build high-quality alpha mask
landmarks = tgt_faces[0].landmark_3d_68[:, :2].astype(np.int32)
mask = np.zeros(target_frame.shape[:2], dtype=np.uint8)
cv2.fillConvexPoly(mask, cv2.convexHull(landmarks), 255)

# Light feathering only (no blurring)
mask = cv2.GaussianBlur(mask, (15, 15), 5)

# Blend swapped face cleanly
alpha = mask[..., None] / 255.0
output = (swapped_img.astype(np.float32) * alpha +
          target_frame.astype(np.float32) * (1.0 - alpha))
output = np.clip(output, 0, 255).astype(np.uint8)

# ✨ Optional sharpness boost (not smoothing!)
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
output = cv2.filter2D(output, -1, sharpen_kernel)

# Save final image with max quality
cv2.imwrite('ironman_is_back.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 100])
print("✅ High-quality swap saved as ironman_is_back.jpg")

# Show result
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Final HQ Output")
plt.show()
