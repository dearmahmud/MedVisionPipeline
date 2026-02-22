# task3_retrieval/evaluate.py
import os
import numpy as np
import faiss

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
ARTIFACT_DIR = "task3_retrieval/artifacts"
EMB_FILE = os.path.join(ARTIFACT_DIR, "embeddings.npy")
LABEL_FILE = os.path.join(ARTIFACT_DIR, "labels.npy")
PATHS_FILE = os.path.join(ARTIFACT_DIR, "paths.txt")
INDEX_FILE = os.path.join(ARTIFACT_DIR, "index.faiss")
TOP_K = [1, 5, 10]


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def load_paths(paths_file):
    with open(paths_file, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def precision_at_k(retrieved_labels, query_label, k):
    if len(retrieved_labels) == 0:
        return 0.0
    k_eff = min(k, len(retrieved_labels))
    return float(np.sum(retrieved_labels[:k_eff] == query_label)) / k_eff


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
def evaluate():
    print("=== Task 3: Semantic Image Retrieval Evaluation ===")

    for f in [EMB_FILE, LABEL_FILE, PATHS_FILE, INDEX_FILE]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing artifact: {f}")

    embeddings = np.load(EMB_FILE).astype("float32")  # [N, D]
    labels = np.load(LABEL_FILE)                      # [N]
    paths = load_paths(PATHS_FILE)                    # [N]
    index = faiss.read_index(INDEX_FILE)

    if len(paths) != embeddings.shape[0] or len(labels) != embeddings.shape[0]:
        raise RuntimeError("Artifact size mismatch among embeddings/labels/paths")

    n = len(embeddings)
    print(f"Loaded {n} embeddings")
    print("Running retrieval evaluation...")

    results = {k: [] for k in TOP_K}

    for i in range(n):
        # ask FAISS for up to n results and remove self
        _, idx = index.search(embeddings[i : i + 1], min(n, max(TOP_K) + 1))
        idx = idx[0]
        idx = idx[idx != i]  # drop self
        idx = idx[idx >= 0]  # drop invalid

        retrieved_labels = labels[idx]

        for k in TOP_K:
            p = precision_at_k(retrieved_labels, query_label=labels[i], k=k)
            results[k].append(p)

    print("\n=== Precision@K Results ===")
    for k in TOP_K:
        mean_p = float(np.mean(results[k])) if len(results[k]) else 0.0
        print(f"Precision@{k}: {mean_p:.4f}")

    print("\nTask 3 evaluation completed successfully.")


if __name__ == "__main__":
    evaluate()