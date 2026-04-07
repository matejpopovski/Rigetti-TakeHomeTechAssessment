"""
Rigetti CV Internship Assessment
---------------------------------
Practical defect classification pipeline for synthetic industrial metal surface images.

Classes: crack, hole, rust, scratch, normal
Architecture: Fine-tuned MobileNetV2 (lightweight, fast, strong baseline)
Framework: PyTorch + torchvision
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib
matplotlib.use("Agg")  # non interactive backend, safe on all platforms
import matplotlib.pyplot as plt

# Constants

CLASSES = ["crack", "hole", "normal", "rust", "scratch"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

IMG_SIZE = 224          # MobileNetV2 expects 224x224
BATCH_SIZE = 64
NUM_EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"   # Apple Silicon GPU
else:
    DEVICE = "cpu"

# Reproducibility

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Dataset

class DefectDataset(Dataset):
    """
    Loads images from dataset/<split>/<class>/<image>.png folder structure.
    Returns (image_tensor, label_int, relative_path_str) per sample.
    Relative paths are relative to data_dir so predictions.csv is portable.
    """

    def __init__(self, root: Path, split: str, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for class_name in CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"  [warn] Missing class directory: {class_dir}", file=sys.stderr)
                continue
            label = CLASS_TO_IDX[class_name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    self.samples.append((img_path, label))

        print(f"  [{split}] {len(self.samples)} images loaded across {len(CLASSES)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # Return path relative to dataset root for portable predictions.csv
        rel_path = img_path.relative_to(self.root)
        return img, label, str(rel_path)

# Transforms

def get_transforms(split: str):
    """
    Training: light augmentation to regularize without distorting defect patterns.
    Test: deterministic resize + normalize only.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

# Model

def build_model(num_classes: int = 5) -> nn.Module:
    """
    MobileNetV2 pre-trained on ImageNet, classifier head replaced for our task.
    Lightweight (~3.4M params) and fast -- well-suited for a 5-class baseline.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    return model

# Training loop

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total

# Evaluation & output

@torch.no_grad()
def run_inference(model, loader, device):
    """Returns (all_relative_paths, all_preds, all_labels)."""
    model.eval()
    all_paths, all_preds, all_labels = [], [], []

    for imgs, labels, paths in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_paths.extend(paths)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    return all_paths, all_preds, all_labels


def save_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(CLASSES)),
        yticks=range(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    thresh = cm.max() / 2.0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved  -> {out_path}")


def print_and_save_evaluation(y_true, y_pred, out_dir: Path):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    report = classification_report(y_true, y_pred, target_names=CLASSES)

    print(f"\n{'='*60}")
    print(f"  Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*60}")

    print("\nPer-class Accuracy:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:<10}: {per_class_acc[i]:.4f}  ({int(cm.diagonal()[i])}/{int(cm.sum(axis=1)[i])})")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = f"{'':>10}" + "".join(f"{c:>10}" for c in CLASSES)
    print(header)
    for i, cls in enumerate(CLASSES):
        row = f"  {cls:<8}" + "".join(f"{cm[i][j]:>10}" for j in range(len(CLASSES)))
        print(row)

    print(f"\nClassification Report:\n{report}")

    # Save confusion matrix image
    save_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png")

    # Save classification report as text
    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)\n\n")
        f.write("Per-class Accuracy:\n")
        for i, cls in enumerate(CLASSES):
            f.write(f"  {cls:<10}: {per_class_acc[i]:.4f}  ({int(cm.diagonal()[i])}/{int(cm.sum(axis=1)[i])})\n")
        f.write(f"\nClassification Report:\n{report}")
    print(f"Classification report saved -> {report_path}")


def save_predictions(paths, preds, out_path: Path):
    """
    Saves predictions.csv with portable relative paths.
    Format: image_path,predicted_label
    e.g.:   test/crack/crack_00001.png,crack
    """
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "predicted_label"])
        for p, pred in zip(paths, preds):
            writer.writerow([p, IDX_TO_CLASS[pred]])
    print(f"Predictions saved       -> {out_path}")

# Main

def parse_args():
    parser = argparse.ArgumentParser(description="Defect image classifier -- Rigetti CV Assessment")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset"),
                        help="Root dataset directory (default: ./dataset)")
    parser.add_argument("--output_dir", type=Path, default=Path("."),
                        help="Where to save outputs (default: .)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training and load existing checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "model_checkpoint.pth"

    print(f"\nDevice : {DEVICE}")
    print(f"Seed   : {args.seed}")
    print(f"Data   : {args.data_dir.resolve()}")
    print(f"Output : {args.output_dir.resolve()}\n")

    print("Loading datasets...")
    train_ds = DefectDataset(args.data_dir, "train", transform=get_transforms("train"))
    test_ds  = DefectDataset(args.data_dir, "test",  transform=get_transforms("test"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=(DEVICE == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=(DEVICE == "cuda"))

    # Model
    model = build_model(num_classes=len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train
    if not args.skip_train:
        print(f"\nTraining MobileNetV2 for {args.epochs} epochs...")
        best_val_acc = 0.0
        history = []

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss,   val_acc   = eval_epoch(model, test_loader,  criterion, DEVICE)
            scheduler.step()

            elapsed = time.time() - t0
            print(f"  Epoch {epoch:>2}/{args.epochs}  "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                  f"({elapsed:.1f}s)")

            history.append({"epoch": epoch, "train_loss": train_loss,
                             "train_acc": train_acc, "val_loss": val_loss,
                             "val_acc": val_acc})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), checkpoint_path)
                print(f"    Checkpoint saved (val_acc={val_acc:.4f})")

        with open(args.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nBest accuracy during training: {best_val_acc:.4f}")

    print(f"\nLoading best checkpoint from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    # Evaluate & save all outputs
    print("\nRunning inference on test set...")
    paths, preds, labels = run_inference(model, test_loader, DEVICE)

    print_and_save_evaluation(labels, preds, args.output_dir)
    save_predictions(paths, preds, args.output_dir / "predictions.csv")


if __name__ == "__main__":
    main()