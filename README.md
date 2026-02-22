# PneumoniaMNIST – Classification, Report Generation, and Semantic Retrieval

This repo implements three connected tasks on MedMNIST v2 PneumoniaMNIST:  
1) CNN classification with rigorous evaluation,  
2) Medical report generation with a visual language model,  
3) Semantic image retrieval with FAISS.

## Quickstart (Colab)

1. Open `main.ipynb` in Google Colab.  
2. Mount Drive and `cd` to the repo root.  
3. **Authenticate Hugging Face** via environment variable or `login()` (use only one).  
4. Run the sequential cells:
   - Task 1 train → evaluate → Task‑1 report  
   - Task 2 generate ≥10 reports → Task‑2 markdown  
   - Task 3 build MedMNIST test index → evaluate P@K → Task‑3 report

Outputs are written to:
- `outputs/` – training curves, ROC, confusion matrices (Task 1)
- `reports/task1_*` – failure CSV + grid, `task1_classification_report.md`
- `reports/task2/samples` – VLM reports per image, `task2_report_generation.md`
- `task3_retrieval/artifacts` – embeddings, labels, paths, FAISS index
- `reports/task3/*` – retrieval CSV + contact sheets, `task3_retrieval_system.md`

## Tasks

### Task 1 – Classification
- Model: ResNet‑18 with 1‑channel conv1, class‑weighted cross‑entropy.
- Data: MedMNIST v2 PneumoniaMNIST train/val/test; grayscale 28×28.
- Scripts:
  - `python -m task1_classification.train --epochs 10 --lr 1e-4 --batch_size 128 --save_path models/resnet_best.pth`
  - `python -m task1_classification.evaluate`
  - `python -m reports.make_task1_report`  
- Report: `reports/task1_classification_report.md` with metrics, ROC/AUC, confusion matrices, failure analysis.

### Task 2 – Report Generation
- Model: `google/medgemma-4b-it` (Hugging Face).
- Pipeline: image + structured prompts, greedy decoding, safety‑first fallbacks.
- Scripts:
  - `python -m reports.make_task2_report` (ensures ≥10 images and saves text outputs)
- Report: `reports/task2_report_generation.md` with prompt strategy notes and samples.

### Task 3 – Semantic Retrieval
- Embeddings: CLIP vision tower pooled + visual projection; L2‑norm; cosine via `IndexFlatIP`.
- Index: built on MedMNIST **test** split with ground‑truth labels.
- Scripts:
  - `python -m task3_retrieval.build_index_medmnist`
  - `python -m task3_retrieval.evaluate`
  - `python -m task3_retrieval.search --query_idx 0 --top_k 5`
  - `python -m task3_retrieval.search --query_text "right lower lobe pneumonia" --top_k 10`
  - `python -m reports.make_task3_report` (P@K summary, CSV, contact sheets)

## Environment