import os
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

os.makedirs("models", exist_ok=True)

print("🔍 Downloading buffalo_l model...")
face_app = FaceAnalysis(name="buffalo_l", root="models")
face_app.prepare(ctx_id=0)

print("✅ buffalo_l downloaded to models/")

print("🔍 Downloading inswapper_128.onnx...")
model = get_model("https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx")
model_path = os.path.join("models", "inswapper_128.onnx")
with open(model_path, "wb") as f:
    f.write(model.session._sess.model.SerializeToString())

print(f"✅ inswapper_128.onnx saved to {model_path}")
