# Task 1 – Pneumonia Classification Report

**Model:** ResNet18 (1‑channel first conv) with class‑weighted CE, Adam, ReduceLROnPlateau. **Dataset:** MedMNIST v2 – PneumoniaMNIST (28×28 grayscale). **Normalization:** mean=0.5, std=0.5. **Augmentations:** horizontal flip, small rotations.

## Metrics and Figures
- ROC curve: `outputs/roc_curve.png`
- Confusion matrix (0.5 threshold): `outputs/confusion_matrix_default.png`
- Confusion matrix (optimal threshold): `outputs/confusion_matrix_optimal.png`
- Failure-case grid (no Grad‑CAM): `outputs/failures_grid.png`

## Failure Case Notes
Misclassifications concentrate on low‑contrast or borderline cases. Optimal thresholding (Youden index) improves balance between sensitivity and specificity.
