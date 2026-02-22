# task1_classification/failure_analysis.py

import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_failures(images, labels, preds, max_examples=10):

    os.makedirs("outputs", exist_ok=True)

    failures = []
    for img, label, pred in zip(images, labels, preds):
        if label != pred:
            failures.append((img[0], label, pred))

    print(f"Total failures: {len(failures)}")

    for i, (img, label, pred) in enumerate(failures[:max_examples]):
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.title(f"True: {label} | Pred: {pred}")
        plt.axis("off")
        plt.savefig(f"outputs/failure_{i}.png")
        plt.close()

    print("Failure cases saved to outputs/")