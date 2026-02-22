# task3_retrieval/build_index_medmnist.py
import os
import numpy as np
import torch
import faiss
from tqdm import tqdm
from PIL import Image
from medmnist import PneumoniaMNIST
from transformers import CLIPModel, CLIPImageProcessor

# ---------------- CONFIG ----------------
ARTIFACT_DIR = "task3_retrieval/artifacts"
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(ARTIFACT_DIR, exist_ok=True)
EMB_FILE   = os.path.join(ARTIFACT_DIR, "embeddings.npy")
LABEL_FILE = os.path.join(ARTIFACT_DIR, "labels.npy")
PATHS_FILE = os.path.join(ARTIFACT_DIR, "paths.txt")
INDEX_FILE = os.path.join(ARTIFACT_DIR, "index.faiss")

def load_model(use_fast: bool = False):
    print("Loading CLIP model...")
    image_proc = CLIPImageProcessor.from_pretrained(MODEL_NAME, use_fast=use_fast)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return model, image_proc

def to_rgb(img):
    # Ensure RGB for CLIP
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    # Handle numpy/torch input robustly (MedMNIST returns PIL by default)
    if torch.is_tensor(img):
        arr = img.squeeze().detach().cpu().numpy()
    else:
        arr = np.array(img).squeeze()
    if arr.dtype != np.uint8:
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")

def main():
    print("=== Task 3: Building Retrieval Index from MedMNIST TEST ===")
    model, image_proc = load_model(use_fast=False)

    test_set = PneumoniaMNIST(split="test", download=True)  # labels are 0/1
    N = len(test_set)
    all_embeddings, all_labels, all_paths = [], [], []

    for i in tqdm(range(0, N, BATCH_SIZE), total=(N + BATCH_SIZE - 1) // BATCH_SIZE):
        batch_imgs, batch_labs = [], []
        j_end = min(i + BATCH_SIZE, N)
        for j in range(i, j_end):
            img, lab = test_set[j]
            batch_imgs.append(to_rgb(img))
            # FIX: avoid NumPy deprecation by extracting scalar safely
            lab_scalar = lab.item() if hasattr(lab, "item") else (lab[0] if isinstance(lab, (list, np.ndarray)) else lab)
            batch_labs.append(int(lab_scalar))

        inputs = image_proc(images=batch_imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            vis = model.vision_model(pixel_values=pixel_values)  # BaseModelOutputWithPooling
            pooled = vis.pooler_output                           # [B, H]
            emb = model.visual_projection(pooled)                # [B, D]
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        all_embeddings.append(emb.cpu().numpy())
        all_labels.extend(batch_labs)
        all_paths.extend([f"medmnist:test_idx={k}" for k in range(i, j_end)])

    embeddings = np.vstack(all_embeddings).astype("float32")  # [N, D]
    labels = np.array(all_labels, dtype=np.int64)             # [N]

    np.save(EMB_FILE, embeddings)
    np.save(LABEL_FILE, labels)
    with open(PATHS_FILE, "w", encoding="utf-8") as f:
        for p in all_paths:
            f.write(p + "\n")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via inner product on L2-normalized vecs
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)

    print("\n✅ Index build completed (MedMNIST test set).")
    print(f"Embeddings: {embeddings.shape}, Labels: {labels.shape}, Index size: {index.ntotal}")

if __name__ == "__main__":
    main()