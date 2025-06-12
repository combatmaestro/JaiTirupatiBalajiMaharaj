import torch
from PIL import Image
from models.e4e_encoder import E4EEncoder
from models.stylegan2_generator import StyleGAN2Generator
import numpy as np
import cv2
from insightface.app import FaceAnalysis

def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)

def get_blended_face(image_list, stylegan_path, e4e_path, use_cuda):
    device = 'cuda' if use_cuda else 'cpu'
    encoder = E4EEncoder(e4e_path, device=device)
    generator = StyleGAN2Generator(stylegan_path, device=device)

    print("📷 Converting input images to PIL...")
    pil_images = []
    for img in image_list:
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pil_images.append(pil_img)
        elif isinstance(img, Image.Image):
            pil_images.append(img)
        else:
            raise ValueError("Unsupported image type in get_blended_face")

    print("🎯 Encoding images into latents...")
    latents = []
    for i, pil_img in enumerate(pil_images):
        print(f"📥 Encoding image {i + 1}/{len(pil_images)}...")
        latent = encoder.encode(pil_img)
        latents.append(latent)

    # ---- New: Select best latent by identity similarity ----
    print("🧠 Finding most identity-preserving latent...")
    face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'] if use_cuda else ['CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0 if use_cuda else -1)

    # Reference face: first input image
    ref_embedding = face_analyzer.get(cv2.cvtColor(np.array(pil_images[0]), cv2.COLOR_RGB2BGR))[0].embedding
    best_idx = 0
    best_score = -1

    for i, latent in enumerate(latents):
        repeated = latent.unsqueeze(1).repeat(1, 18, 1) if latent.ndim == 2 else latent
        generated_img = generator.synthesize(repeated)
        gen_bgr = cv2.cvtColor(np.array(generated_img), cv2.COLOR_RGB2BGR)
        faces = face_analyzer.get(gen_bgr)
        if not faces:
            print(f"❌ No face found in generated latent {i}")
            continue
        emb = faces[0].embedding
        score = cosine_similarity(ref_embedding, emb)
        print(f"🔍 Latent {i} identity similarity score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_idx = i

    print(f"✅ Selected latent {best_idx} with similarity score {best_score:.4f}")
    best_latent = latents[best_idx]
    repeated_latent = best_latent.unsqueeze(1).repeat(1, 18, 1) if best_latent.ndim == 2 else best_latent

    print("🖼️ Synthesizing most identity-preserving face...")
    result_image = generator.synthesize(repeated_latent)
    return cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
