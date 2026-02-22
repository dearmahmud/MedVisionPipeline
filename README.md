# MedVisionPipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dearmahmud/MedVisionPipeline/blob/master/notebooks/Colab_main.ipynb)

End‑to‑end pipeline for chest X‑ray analysis across three tasks:

1. **Task‑1 – Classification:** ResNet‑based pneumonia classifier on MedMNIST PneumoniaMNIST (28×28 grayscale).
2. **Task‑2 – Report Generation:** Structured radiology reports (Findings / Impression / Pneumonia yes–no).
3. **Task‑3 – Semantic Image Retrieval:** CLIP embeddings + FAISS index; Precision@K + query demos.

Built to **run in Google Colab** and **submit via GitHub**.

---

## Quick Start (Colab)

1. Click the badge above to open the notebook in Colab.  
2. (Recommended) **Runtime → Change runtime type → GPU (T4)**.  
3. **Run all** cells from top to bottom. The notebook:
   - Boots into the repo root in `/content/MedVisionPipeline`
   - Installs `requirements.txt`
   - Creates standard output folders
   - Executes Task‑1 → Task‑2 → Task‑3

> **Note:** Task‑2 may require a Hugging Face token. See **Hugging Face Token (Task‑2)** below.

---

## Quick Start (Local)

> Local runs are possible but slower without CUDA. Python **3.9+** recommended.

```bash
# 1) Clone
git clone https://github.com/dearmahmud/MedVisionPipeline.git
cd MedVisionPipeline

# 2) (Optional) Virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

# 3) Install deps
pip install -r requirements.txt

# 4) Run tasks (examples)
python -m task1_classification.train --epochs 10 --lr 1e-4 --batch_size 128 --save_path models/resnet_best.pth
python -m task1_classification.evaluate --model_path models/resnet_best.pth

python -m reports.make_task1_report
python -m reports.make_task2_report              # requires HF token if using a gated model
python -m task3_retrieval.build_index_medmnist
python -m task3_retrieval.evaluate
python -m reports.make_task3_report
```

---

## Repository Structure

```
MedVisionPipeline/
  notebooks/
    Colab_main.ipynb            # Colab entry point (one‑click)
  task1_classification/         # Training, evaluation, plots, reporting glue
  task2_report_generation/      # Report generation utilities
  task3_retrieval/              # CLIP embeddings, FAISS index, eval, search
  models/                       # Saved weights (created at runtime)
  data/                         # Optional local images; MedMNIST auto-downloads
  reports/                      # Markdown reports + figs/tables (created at runtime)
  outputs/                      # Curves/plots from Task‑1 (created at runtime)
  requirements.txt
  README.md
```

> Large artifacts (`*.pth`, `*.npy`, `*.faiss`) are generated at runtime and not committed.

---

## Tasks

### Task‑1 — Classification

**Train + Evaluate**
```bash
# Train a ResNet‑18 variant (1‑channel first conv) on PneumoniaMNIST
python -m task1_classification.train \
  --epochs 10 \
  --lr 1e-4 \
  --batch_size 128 \
  --save_path models/resnet_best.pth

# Evaluate: saves ROC, confusion matrices, and curves to outputs/
python -m task1_classification.evaluate --model_path models/resnet_best.pth

# Build a concise markdown report (reports/task1/)
python -m reports.make_task1_report
```

**Artifacts**
- `models/resnet_best.pth`
- `outputs/loss_curve.png`, `outputs/val_accuracy_curve.png`, `outputs/roc_curve.png`, confusion matrices
- `reports/task1/...` (markdown, failure‑case grid + CSV)

---

### Task‑2 — Report Generation

Produces structured reports with:
- **Findings**
- **Impression**
- **Pneumonia present (yes/no)**

**Run**
```bash
python -m reports.make_task2_report
```

If the selected VLM is gated on Hugging Face, an access token is required at runtime (details below).  
If no token is supplied, the notebook will **skip Task‑2 gracefully** and complete Tasks 1 & 3.

**Artifacts**
- `reports/task2_report_generation.md`
- `reports/task2/samples/` (per‑image text reports)

---

### Task‑3 — Semantic Image Retrieval

Build CLIP embeddings + FAISS index on the **PneumoniaMNIST test split**, evaluate Precision@K, and create contact sheets.

**Run**
```bash
# Build index from MedMNIST test
python -m task3_retrieval.build_index_medmnist

# Evaluate Precision@{1,5,10}
python -m task3_retrieval.evaluate

# Write markdown report (reports/task3/)
python -m reports.make_task3_report

# Optional demos
python -m task3_retrieval.search --query_idx 0 --top_k 10
python -m task3_retrieval.search --query_text "right lower lobe pneumonia" --top_k 10
```

**Artifacts**
- `task3_retrieval/artifacts/` (embeddings, labels, paths, FAISS index)
- `reports/task3/...` (Precision@K summary, contact‑sheet figures)

---

## Hugging Face Token (Task‑2)

If your chosen model is gated:

**In Colab (preferred)**
1. **Runtime → Variables** → add:
   - Key: `HF_TOKEN`
   - Value: *your token*
2. Run the notebook; it will detect the token automatically.

**Or paste when prompted** (the notebook asks if no token is found).

**No token?**  
Tasks 1 and 3 still run. Task‑2 is skipped automatically without errors.  
Tokens are **not** stored in the repo for security.

---

## Outputs

- **Task‑1:** `outputs/` (curves/plots), `reports/task1/` (markdown + failure analysis)
- **Task‑2:** `reports/task2_report_generation.md`, `reports/task2/samples/` (text)
- **Task‑3:** `task3_retrieval/artifacts/` (embeddings/index), `reports/task3/` (markdown + figures)

---

## Reproducibility

- Fixed random seeds in training where applicable.  
- Deterministic evaluation preprocessing.  
- All paths are **relative** (no `/content/drive/...`).  
- Dependencies listed in `requirements.txt`.

---

## Troubleshooting

- **`ModuleNotFoundError`**: ensure execution from repo root; the Colab notebook bootstraps this.  
- **No CUDA in Colab**: Runtime → Change runtime type → GPU (T4).  
- **HF token errors (Task‑2)**: provide `HF_TOKEN` via Runtime Variables or paste when prompted; otherwise Task‑2 is skipped.  
- **Missing large artifacts**: generated at runtime by the scripts.

---

## Citation & Acknowledgments

- **MedMNIST** (PneumoniaMNIST) dataset.  
- **ResNet‑18** (torchvision).  
- **CLIP** (transformers) and **FAISS** for retrieval.  
- **Hugging Face** Hub/Transformers for VLM inference in Task‑2.

---

## Maintainer

**Dr. Mohammad Mahmudul Hasan**  
Associate Professor  
Department of Electrical and Electronic Engineering  
Email: mohammad.m.hasan@ntnu.no  
Phone: +47 463 45 632  
Website: https://dearmahmud.github.io  

For issues or requests, please open a GitHub issue or contact me.

---

## License

Academic use. Add a `LICENSE` file (e.g., MIT) if a formal license is required.

---

### Submission Links (for evaluators)

- **GitHub Repository:** https://github.com/dearmahmud/MedVisionPipeline  
- **Colab Notebook:** https://colab.research.google.com/github/dearmahmud/MedVisionPipeline/blob/master/notebooks/Colab_main.ipynb
