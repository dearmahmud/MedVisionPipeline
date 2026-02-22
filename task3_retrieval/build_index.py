# task3_retrieval/build_index.py
import os
import glob
import numpy as np
import torch
import faiss
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor

# ---------------- CONFIG ----------------
DATA_DIR = "data/images"
ARTIFACT_DIR = "task3_retrieval/artifacts"
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(ARTIFACT_DIR, exist_ok=True)
EMB_FILE = os.path.join(ARTIFACT_DIR, "embeddings.npy")
LABEL_FILE = os.path.join(ARTIFACT_DIR, "labels.npy")
PATHS_FILE = os.path.join(ARTIFACT_DIR, "paths.txt")
INDEX_FILE = os.path.join(ARTIFACT_DIR, "index.faiss")


# ---------------- LOAD MODEL ----------------
def load_model(use_fast=False):
    print("Loading CLIP model...")
    image_proc = CLIPImageProcessor.from_pretrained(MODEL_NAME, use_fast=use_fast)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return model, image_proc


# ---------------- HELPERS ----------------
def list_images(data_dir):
    paths = sorted(
        glob.glob(os.path.join(data_dir, "*.png"))
        + glob.glob(os.path.join(data_dir, "*.jpg"))
        + glob.glob(os.path.join(data_dir, "*.jpeg"))
    )
    return paths


def save_paths(paths, path_file):
    with open(path_file, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")


# ---------------- MAIN ----------------
def main():
    print("=== Task 3: Building Retrieval Index ===")
    model, image_proc = load_model(use_fast=False)

    image_paths = list_images(DATA_DIR)
    if len(image_paths) == 0:
        raise RuntimeError("❌ No images found in data/images")

    all_embeddings = []
    all_labels = []

    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]

        inputs = image_proc(images=batch_imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            # Vision-only pass to avoid text requirements
            vision_out = model.vision_model(pixel_values=pixel_values)   # BaseModelOutputWithPooling
            pooled = vision_out.pooler_output                            # [B, H]
            emb = model.visual_projection(pooled)                         # [B, D]
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)         # cosine-ready

        all_embeddings.append(emb.cpu().numpy())
        for p in batch_paths:
            # crude label from path name, same as your original logic
            all_labels.append(1 if "pneumonia" in p.lower() else 0)

    embeddings = np.vstack(all_embeddings).astype("float32")  # [N, D]
    labels = np.array(all_labels, dtype=np.int64)

    # Save artifacts
    np.save(EMB_FILE, embeddings)
    np.save(LABEL_FILE, labels)
    save_paths(image_paths, PATHS_FILE)

    # FAISS index: cosine similarity == inner product after L2 norm
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)

    print("\n✅ Index build completed successfully!")
    print(f"Embeddings: {embeddings.shape}, Labels: {labels.shape}, Index size: {index.ntotal}")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    main()