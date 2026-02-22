# reports/make_task3_report.py
import os
import io
import re
import numpy as np
import faiss
from PIL import Image, ImageDraw, ImageFont
from medmnist import PneumoniaMNIST

ART_DIR = "task3_retrieval/artifacts"
EMB_PATH = os.path.join(ART_DIR, "embeddings.npy")
LAB_PATH = os.path.join(ART_DIR, "labels.npy")
PATHS_PATH = os.path.join(ART_DIR, "paths.txt")
INDEX_PATH = os.path.join(ART_DIR, "index.faiss")

OUT_DIR = "reports/task3"
FIG_DIR = os.path.join(OUT_DIR, "figures")
CSV_DIR = os.path.join(OUT_DIR, "tables")
MD_PATH = os.path.join("reports", "task3_retrieval_system.md")

TOPK_LIST = [1, 5, 10]
CONTACT_K = 5
CANVAS_W = 1200
TILE = 224
PAD = 8

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

def _load_paths():
    with open(PATHS_PATH, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def _parse_medmnist_index(tag: str) -> int:
    # tag like "medmnist:test_idx=167"
    m = re.search(r"test_idx=(\d+)", tag)
    return int(m.group(1)) if m else -1

def _get_pil_from_test(idx: int) -> Image.Image:
    ds = PneumoniaMNIST(split="test", download=True)
    img, lab = ds[idx]
    if not isinstance(img, Image.Image):
        img = Image.fromarray((np.array(img).squeeze()*255).astype("uint8"))
    return img.convert("RGB")

def _contact_sheet(query_img: Image.Image, neighbors: list, title: str, out_path: str):
    # neighbors: list of tuples (pil_img, score, label)
    cols = CONTACT_K + 1  # query + top-k
    w = PAD + cols*(TILE + PAD)
    h = PAD*2 + TILE + 40
    sheet = Image.new("RGB", (max(w, CANVAS_W), h), (255,255,255))
    draw = ImageDraw.Draw(sheet)

    # Paste query
    q = query_img.resize((TILE, TILE))
    sheet.paste(q, (PAD, PAD))

    # Paste neighbors
    for i, (img, score, lab) in enumerate(neighbors[:CONTACT_K], start=1):
        p = img.resize((TILE, TILE))
        x = PAD + i*(TILE + PAD)
        sheet.paste(p, (x, PAD))
        draw.text((x, PAD + TILE + 5), f"{score:.3f} | y={lab}", fill=(0,0,0))

    # Title
    draw.text((PAD, TILE + PAD + 24), title, fill=(0,0,0))
    sheet.save(out_path)

def evaluate_p_at_k(X: np.ndarray, y: np.ndarray, index, ks):
    n = len(X)
    res = {k: [] for k in ks}
    maxK = min(n, max(ks) + 1)
    for i in range(n):
        D, I = index.search(X[i:i+1], maxK)
        I = I[0]
        I = I[I != i]
        I = I[I >= 0]
        retrieved = y[I]
        for k in ks:
            kk = min(k, len(retrieved))
            if kk == 0:
                res[k].append(0.0)
            else:
                res[k].append(float(np.sum(retrieved[:kk] == y[i])) / kk)
    return {k: float(np.mean(res[k])) if len(res[k]) else 0.0 for k in ks}

def main():
    if not all(os.path.exists(p) for p in [EMB_PATH, LAB_PATH, PATHS_PATH, INDEX_PATH]):
        raise FileNotFoundError("Missing Task-3 artifacts. Build index first.")

    X = np.load(EMB_PATH).astype("float32")
    y = np.load(LAB_PATH)
    paths = _load_paths()
    index = faiss.read_index(INDEX_PATH)

    # P@K table to markdown
    p_at_k = evaluate_p_at_k(X, y, index, TOPK_LIST)
    md = io.StringIO()
    md.write("# Task 3 – Semantic Image Retrieval\n\n")
    md.write("**Embeddings:** CLIP vision tower (vit‑b/32) pooled + projection, L2‑normalized. **Index:** FAISS inner‑product (cosine). **Dataset:** MedMNIST PneumoniaMNIST test split.\n\n")
    md.write("## Precision@K\n")
    for k in TOPK_LIST:
        md.write(f"- Precision@{k}: {p_at_k[k]:.4f}\n")

    # CSV log for a few queries
    csv_path = os.path.join(CSV_DIR, "retrieval_log.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("query_idx,rank,neighbor_idx,score,label\n")
        for q in [0, 10, 50, 123]:
            if q >= len(X): 
                continue
            D, I = index.search(X[q:q+1], min(len(X), CONTACT_K + 1))
            I, D = I[0], D[0]
            mask = I != q
            I, D = I[mask], D[mask]
            for r, (j, s) in enumerate(zip(I[:CONTACT_K], D[:CONTACT_K]), start=1):
                f.write(f"{q},{r},{j},{float(s):.6f},{int(y[j])}\n")

    # Contact sheets for image->image
    for q in [0, 10]:
        if q >= len(X):
            continue
        # Query PIL
        qi = _parse_medmnist_index(paths[q])
        qimg = _get_pil_from_test(qi)
        D, I = index.search(X[q:q+1], min(len(X), CONTACT_K + 1))
        I, D = I[0], D[0]
        mask = I != q
        I, D = I[mask], D[mask]
        neigh = []
        for j, s in zip(I[:CONTACT_K], D[:CONTACT_K]):
            jj = _parse_medmnist_index(paths[j])
            neigh.append((_get_pil_from_test(jj), float(s), int(y[j])))
        fig_path = os.path.join(FIG_DIR, f"contact_image_q{q}.png")
        _contact_sheet(qimg, neigh, f"Query idx={q} y={int(y[q])}", fig_path)

    # Contact sheet for a text query
    # We cannot compute text embeddings here without loading CLIP; this file focuses on reporting.
    # The search CLI already supports --query_text. Here, just document the CLI usage.
    md.write("\n## Artifacts\n")
    md.write(f"- Retrieval CSV: `{csv_path}`\n")
    for q in [0, 10]:
        fig_path = os.path.join(FIG_DIR, f"contact_image_q{q}.png")
        if os.path.exists(fig_path):
            md.write(f"- Contact sheet (image query {q}): `{fig_path}`\n")

    md.write("\n## Notes on text→image\n")
    md.write("Text→image works mechanically with CLIP text tower, but clinical phrases on 28×28 CXRs align weakly. Stronger alignment requires medical CLIP variants or prompt ensembling.\n")

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md.getvalue())

    print(f"Wrote {MD_PATH}")

if __name__ == "__main__":
    main()