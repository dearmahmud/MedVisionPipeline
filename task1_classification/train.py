# task1_classification/train.py

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.dataset import get_dataloaders
from models.resnet_model import ResNetPneumonia


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Training Function
# -----------------------------
def train(num_epochs, lr, batch_size, save_path):

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    model = ResNetPneumonia(pretrained=True).to(device)

    # Compute class weights dynamically
    class_counts = [0, 0]
    for _, labels in train_loader:
        for l in labels.squeeze():
            class_counts[int(l)] += 1

    total = sum(class_counts)
    class_weights = [total / c for c in class_counts]

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler (compatible with older PyTorch)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")

    for epoch in range(num_epochs):

        # ----------------- TRAIN -----------------
        model.train()
        running_loss = 0

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        # ----------------- VALIDATE -----------------
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.squeeze().long().to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("✅ Best model saved.")

    # Save training history
    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/training_history.npy", history)

    # Plot training curves
    plot_training_curves(history)


# -----------------------------
# Plot Curves
# -----------------------------
def plot_training_curves(history):

    os.makedirs("outputs", exist_ok=True)

    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.savefig("outputs/loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.savefig("outputs/val_accuracy_curve.png")
    plt.close()

    print("📊 Training curves saved to outputs/")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_path", type=str, default="models/resnet_best.pth")

    args = parser.parse_args()

    train(
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        save_path=args.save_path
    )