# task1_classification/evaluate.py
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from data.dataset import get_dataloaders
from models.resnet_model import ResNetPneumonia

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def _to_numpy(x):
    return x.detach().cpu().numpy()

def _compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(y_true, y_pred)
    return dict(acc=acc, prec=pr, rec=rc, f1=f1, auc=auc, thr=thr, cm=cm), y_pred

def _plot_roc(y_true, y_prob, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', lw=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve (Task 1)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def _plot_cm(cm, labels, path, title):
    import seaborn as sns
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def evaluate(model_path="models/resnet_best.pth", batch_size=128, save_images_grid=True, grid_max=24):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders(batch_size=batch_size)

    model = ResNetPneumonia(pretrained=False).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing weights: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    xs, ys, probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)           # [B,1,28,28]
            labels = labels.squeeze().long()     # CPU ok for metrics later
            logits = model(images)
            p1 = F.softmax(logits, dim=1)[:, 1]
            xs.append(images.cpu())
            ys.append(labels.cpu())
            probs.append(p1.cpu())

    X = torch.cat(xs, dim=0)                    # [N,1,28,28]
    Y = torch.cat(ys, dim=0).numpy()            # [N]
    P = torch.cat(probs, dim=0).numpy()         # [N], prob of class=1

    # Metrics @ 0.5
    m_default, y_pred_default = _compute_metrics(Y, P, thr=0.5)

    # Optimal threshold via Youden index
    fpr, tpr, thr = roc_curve(Y, P)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    best_thr = float(thr[best_idx]) if best_idx < len(thr) else 0.5
    m_opt, y_pred_opt = _compute_metrics(Y, P, thr=best_thr)

    # Plots
    _plot_roc(Y, P, os.path.join(OUT_DIR, "roc_curve.png"))
    _plot_cm(m_default["cm"], ["normal(0)", "pneum(1)"],
             os.path.join(OUT_DIR, "confusion_matrix_default.png"),
             "Confusion Matrix (thr=0.5)")
    _plot_cm(m_opt["cm"], ["normal(0)", "pneum(1)"],
             os.path.join(OUT_DIR, "confusion_matrix_optimal.png"),
             f"Confusion Matrix (thr={best_thr:.3f})")

    # Optional: save a failure-case grid (raw images, no Grad-CAM)
    grid_path = None
    if save_images_grid:
        wrong_idx = np.where(Y != y_pred_default)[0]
        grid_idx = wrong_idx[:grid_max]
        import math
        cols = 6
        rows = max(1, math.ceil(len(grid_idx) / cols))
        if len(grid_idx) == 0:
            # create placeholder
            plt.figure(figsize=(6, 2))
            plt.text(0.5, 0.5, "No misclassifications on the test set.",
                     ha="center", va="center")
            plt.axis("off")
            grid_path = os.path.join(OUT_DIR, "failures_grid.png")
            plt.tight_layout(); plt.savefig(grid_path, dpi=150); plt.close()
        else:
            fig, axes = plt.subplots(rows, cols, figsize=(2.3*cols, 2.3*rows))
            axes = np.array(axes).reshape(rows, cols)
            for k, idx in enumerate(grid_idx):
                ax = axes[k // cols, k % cols]
                img = X[idx].squeeze().numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                ax.imshow(img, cmap="gray")
                ax.set_title(f"i={idx} y={Y[idx]} p1={P[idx]:.2f}", fontsize=8)
                ax.axis("off")
            for k in range(len(grid_idx), rows*cols):
                axes[k // cols, k % cols].axis("off")
            grid_path = os.path.join(OUT_DIR, "failures_grid.png")
            fig.tight_layout(); fig.savefig(grid_path, dpi=150); plt.close(fig)

    # Text report to console
    print("\n=== Task 1 Metrics (thr=0.5) ===")
    print(f"ACC={m_default['acc']:.4f}  P={m_default['prec']:.4f}  R={m_default['rec']:.4f}  F1={m_default['f1']:.4f}  AUC={m_default['auc']:.4f}")
    print("\n=== Task 1 Metrics (optimal thr) ===")
    print(f"thr={best_thr:.4f}  ACC={m_opt['acc']:.4f}  P={m_opt['prec']:.4f}  R={m_opt['rec']:.4f}  F1={m_opt['f1']:.4f}  AUC={m_opt['auc']:.4f}")

    # Return for downstream use
    return X, Y, y_pred_default

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="models/resnet_best.pth")
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()
    evaluate(model_path=args.model_path, batch_size=args.batch_size)