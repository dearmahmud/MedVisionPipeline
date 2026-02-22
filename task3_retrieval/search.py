# task3_retrieval/search.py
import os
import argparse
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer

# ---------------- CONFIG ----------------
ARTIFACT_DIR = "task3_retrieval/artifacts"
EMB_FILE   = os.path.join(ARTIFACT_DIR, "embeddings.npy")
LABEL_FILE = os.path.join(ARTIFACT_DIR, "labels.npy")
PATHS_FILE = os.path.join(ARTIFACT_DIR, "paths.txt")
INDEX_FILE = os.path.join(ARTIFACT_DIR, "index.faiss")
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NEG_INF = -3.4028234663852886e38  # FAISS float32 sentinel

def load_paths(path_file):
    with open(path_file, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def load_model(use_fast=False):
    img_proc = CLIPImageProcessor.from_pretrained(MODEL_NAME, use_fast=use_fast)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return model, img_proc, tokenizer

def encode_image(model, img_proc, image_path):
    img = Image.open(image_path).convert("RGB")
    inputs = img_proc(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)
    with torch.no_grad():
        vis = model.vision_model(pixel_values=pixel_values)
        pooled = vis.pooler_output
        emb = model.visual_projection(pooled)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb.cpu().numpy().astype("float32")  # [1, D]

def encode_text(model, tokenizer, text):
    tokens = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        text_out = model.text_model(**tokens)        # BaseModelOutputWithPooling
        pooled = text_out.pooler_output              # [B, H]
        emb = model.text_projection(pooled)          # [B, D]
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb.cpu().numpy().astype("float32")  # [1, D]

def _clean(I, D):
    seen = set()
    out_i, out_d = [], []
    for j, s in zip(I, D):
        if j < 0:             # FAISS no-hit id
            continue
        if s <= NEG_INF / 2:  # guard for sentinel
            continue
        if j in seen:
            continue
        seen.add(j)
        out_i.append(j); out_d.append(s)
    return np.array(out_i, dtype=np.int64), np.array(out_d, dtype=np.float32)

def search_index(query_idx, X, y, paths, index, top_k):
    n = index.ntotal
    k_eff = max(0, min(top_k, n - 1))
    if k_eff == 0:
        return []
    D, I = index.search(X[query_idx:query_idx+1], k_eff + 1)
    I, D = I[0], D[0]
    mask = I != query_idx
    I, D = I[mask], D[mask]
    I, D = _clean(I, D)
    I, D = I[:k_eff], D[:k_eff]
    return [{"rank": r+1, "path": paths[j], "label": int(y[j]), "score_ip": float(s)} for r, (j, s) in enumerate(zip(I, D))]

def search_vector(q, y, paths, index, top_k):
    n = index.ntotal
    k_eff = max(0, min(top_k, n))
    if k_eff == 0:
        return []
    D, I = index.search(q, k_eff)
    I, D = I[0], D[0]
    I, D = _clean(I, D)
    I, D = I[:k_eff], D[:k_eff]
    return [{"rank": r+1, "path": paths[j], "label": int(y[j]), "score_ip": float(s)} for r, (j, s) in enumerate(zip(I, D))]

def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--query_idx",  type=int)
    g.add_argument("--query_path", type=str)
    g.add_argument("--query_text", type=str)
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()

    for f in [EMB_FILE, LABEL_FILE, PATHS_FILE, INDEX_FILE]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing artifact: {f}")

    X = np.load(EMB_FILE).astype("float32")  # [N, D]
    y = np.load(LABEL_FILE)                  # [N]
    paths = load_paths(PATHS_FILE)
    index = faiss.read_index(INDEX_FILE)

    if len(paths) != X.shape[0] or len(y) != X.shape[0]:
        raise RuntimeError("Artifact size mismatch among embeddings/labels/paths")

    if args.query_idx is not None:
        if not (0 <= args.query_idx < X.shape[0]):
            raise ValueError(f"query_idx out of range [0, {X.shape[0]-1}]")
        results = search_index(args.query_idx, X, y, paths, index, args.top_k)
        print(f"Query by index = {args.query_idx} -> {paths[args.query_idx]}")
    elif args.query_path is not None:
        model, img_proc, tok = load_model(use_fast=False)
        q = encode_image(model, img_proc, args.query_path)
        results = search_vector(q, y, paths, index, args.top_k)
        print(f"Query by path = {args.query_path}")
    else:
        model, img_proc, tok = load_model(use_fast=False)
        q = encode_text(model, tok, args.query_text)
        results = search_vector(q, y, paths, index, args.top_k)
        print(f"Query by text = {args.query_text!r}")

    print("\nTop-K results:")
    for r in results:
        print(f"[{r['rank']}] score={r['score_ip']:.4f}  label={r['label']}  path={r['path']}")

if __name__ == "__main__":
    main()