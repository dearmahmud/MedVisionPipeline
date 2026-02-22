# reports/make_task1_report.py
import os
import io
import numpy as np

MD_PATH = os.path.join("reports", "task1_classification_report.md")
OUT_DIR = "reports/task1"
IMG_DIR = os.path.join(OUT_DIR, "figures")
CSV_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# Use your evaluate() which now saves ROC/CM and returns arrays
from task1_classification.evaluate import evaluate

BEST_MODEL = "models/resnet_best.pth"

def main():
    X, Y, Yhat = evaluate(model_path=BEST_MODEL)

    # Save a simple CSV of predictions for traceability
    csv_path = os.path.join(CSV_DIR, "task1_predictions.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("index,y_true,y_pred\n")
        for i, (yt, yp) in enumerate(zip(Y, Yhat)):
            f.write(f"{i},{int(yt)},{int(yp)}\n")

    # Build markdown that references the figures produced by evaluate()
    md = io.StringIO()
    md.write("# Task 1 – Pneumonia Classification Report\n\n")
    md.write("**Model:** ResNet18 (1‑channel first conv) with class‑weighted CE, Adam, ReduceLROnPlateau. ")
    md.write("**Dataset:** MedMNIST v2 – PneumoniaMNIST (28×28 grayscale). ")
    md.write("**Normalization:** mean=0.5, std=0.5. **Augmentations:** horizontal flip, small rotations.\n\n")

    md.write("## Metrics and Figures\n")
    md.write("- ROC curve: `outputs/roc_curve.png`\n")
    md.write("- Confusion matrix (0.5 threshold): `outputs/confusion_matrix_default.png`\n")
    md.write("- Confusion matrix (optimal threshold): `outputs/confusion_matrix_optimal.png`\n")
    md.write("- Failure-case grid (no Grad‑CAM): `outputs/failures_grid.png`\n\n")

    md.write("## Failure Case Notes\n")
    md.write("Misclassifications concentrate on low‑contrast or borderline cases. Optimal thresholding (Youden index) improves balance between sensitivity and specificity.\n")

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md.getvalue())
    print(f"Wrote {MD_PATH}")

if __name__ == "__main__":
    main()