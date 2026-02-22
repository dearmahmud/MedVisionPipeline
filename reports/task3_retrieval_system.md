# Task 3 – Semantic Image Retrieval

**Embeddings:** CLIP vision tower (vit‑b/32) pooled + projection, L2‑normalized. **Index:** FAISS inner‑product (cosine). **Dataset:** MedMNIST PneumoniaMNIST test split.

## Precision@K
- Precision@1: 0.8029
- Precision@5: 0.7833
- Precision@10: 0.7761

## Artifacts
- Retrieval CSV: `reports/task3/tables/retrieval_log.csv`
- Contact sheet (image query 0): `reports/task3/figures/contact_image_q0.png`
- Contact sheet (image query 10): `reports/task3/figures/contact_image_q10.png`

## Notes on text→image
Text→image works mechanically with CLIP text tower, but clinical phrases on 28×28 CXRs align weakly. Stronger alignment requires medical CLIP variants or prompt ensembling.
