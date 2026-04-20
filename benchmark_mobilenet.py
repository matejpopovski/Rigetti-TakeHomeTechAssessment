"""
benchmark_mobilenet.py

Compares MobileNetV2 vs MobileNetV3-Small on the Rigetti defect dataset.
Measures training time and validation accuracy for 1, 3, and 5 epochs.
Saves a benchmark summary and prints a conclusion.
"""

import argparse
import csv
import json
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
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Constants
# -----------------------------
CLASSES = ["crack", "hole", "normal", "rust", "scratch"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

IMG_SIZE = 224
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42
EPOCH_OPTIONS = [1, 3, 5]

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset
# -----------------------------
class DefectDataset(Dataset):
    """
    Expects:
        dataset/
            train/
                crack/
                hole/
                normal/
                rust/
                scratch/
            test/
                crack/
                hole/
                normal/
                rust/
                scratch/
    """

    def __init__(self, root: Path, split: str, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = []

        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for class_name in CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"[warn] Missing class directory: {class_dir}", file=sys.stderr)
                continue

            label = CLASS_TO_IDX[class_name]

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    self.samples.append((img_path, label))

        print(f"[{split}] Loaded {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        rel_path = img_path.relative_to(self.root)
        return image, label, str(rel_path)


# -----------------------------
# Transforms
# -----------------------------
def get_transforms(split: str):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])


# -----------------------------
# Model builders
# -----------------------------
def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )
        return model

    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

        # torchvision MobileNetV3 classifier is usually:
        # Sequential(
        #   Linear(...),
        #   Hardswish(),
        #   Dropout(...),
        #   Linear(...),
        # )
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


# -----------------------------
# Train / Eval
# -----------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_seen += images.size(0)

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_seen += images.size(0)

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_paths, all_preds, all_labels = [], [], []

    for images, labels, paths in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_paths.extend(paths)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    return all_paths, all_preds, all_labels


# -----------------------------
# Output helpers
# -----------------------------
def save_predictions(paths, preds, out_path: Path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "predicted_label"])
        for path, pred in zip(paths, preds):
            writer.writerow([path, IDX_TO_CLASS[pred]])


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

    threshold = cm.max() / 2.0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_benchmark_csv(results, out_path: Path):
    fieldnames = ["model", "epochs", "train_time_sec", "val_loss", "val_acc"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def make_conclusion(results):
    lines = []
    lines.append("Benchmark Conclusion")
    lines.append("=" * 60)

    grouped = {}
    for row in results:
        grouped.setdefault(row["model"], []).append(row)

    for model_name, rows in grouped.items():
        rows = sorted(rows, key=lambda x: x["epochs"])
        lines.append(f"\n{model_name}:")
        for row in rows:
            lines.append(
                f"  - {row['epochs']} epoch(s): "
                f"time={row['train_time_sec']:.2f}s, "
                f"val_loss={row['val_loss']:.6f}, "
                f"val_acc={row['val_acc']:.4f}"
            )

    best_acc_run = max(results, key=lambda x: x["val_acc"])
    fastest_run = min(results, key=lambda x: x["train_time_sec"])

    lines.append("\nSummary:")
    lines.append(
        f"  Best validation accuracy: {best_acc_run['model']} "
        f"at {best_acc_run['epochs']} epoch(s) with val_acc={best_acc_run['val_acc']:.4f}"
    )
    lines.append(
        f"  Fastest run: {fastest_run['model']} "
        f"at {fastest_run['epochs']} epoch(s) with train_time={fastest_run['train_time_sec']:.2f}s"
    )

    lines.append("\nInterpretation:")
    lines.append(
        "  Since this is a synthetic defect dataset and both models are pretrained, "
        "it is expected that both MobileNet variants reach high validation accuracy quickly. "
        "If the accuracy difference is very small, the preferred model is usually the one "
        "with the lower training time and simpler deployment profile."
    )

    return "\n".join(lines)


# -----------------------------
# Benchmark core
# -----------------------------
def benchmark_single_model(model_name, train_loader, test_loader, output_dir, epoch_options, lr):
    results = []

    for num_epochs in epoch_options:
        print(f"\nRunning {model_name} for {num_epochs} epoch(s)...")
        set_seed(DEFAULT_SEED)

        model = build_model(model_name, num_classes=len(CLASSES)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        run_start = time.time()
        history = []

        for epoch_idx in range(1, num_epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc = eval_epoch(model, test_loader, criterion, DEVICE)
            scheduler.step()

            epoch_time = time.time() - epoch_start
            print(
                f"  Epoch {epoch_idx:>2}/{num_epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"time={epoch_time:.2f}s"
            )

            history.append({
                "epoch": epoch_idx,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_sec": epoch_time,
            })

        total_train_time = time.time() - run_start
        final_val_loss, final_val_acc = eval_epoch(model, test_loader, criterion, DEVICE)

        results.append({
            "model": model_name,
            "epochs": num_epochs,
            "train_time_sec": round(total_train_time, 2),
            "val_loss": round(final_val_loss, 6),
            "val_acc": round(final_val_acc, 6),
        })

        # Save detailed outputs for the largest epoch run only
        if num_epochs == max(epoch_options):
            model_dir = output_dir / f"{model_name}_{num_epochs}epochs"
            model_dir.mkdir(parents=True, exist_ok=True)

            paths, preds, labels = run_inference(model, test_loader, DEVICE)
            save_predictions(paths, preds, model_dir / "predictions.csv")
            save_confusion_matrix(labels, preds, model_dir / "confusion_matrix.png")

            with open(model_dir / "training_history.json", "w") as f:
                json.dump(history, f, indent=2)

    return results


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MobileNetV2 vs MobileNetV3-Small")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output_dir", type=Path, default=Path("benchmark_outputs"))
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Use 0 on macOS if dataloader multiprocessing is unstable")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDevice : {DEVICE}")
    print(f"Data   : {args.data_dir.resolve()}")
    print(f"Output : {args.output_dir.resolve()}")
    print(f"Batch  : {args.batch_size}")
    print(f"Seed   : {args.seed}\n")

    print("Loading datasets...")
    train_dataset = DefectDataset(args.data_dir, "train", transform=get_transforms("train"))
    test_dataset = DefectDataset(args.data_dir, "test", transform=get_transforms("test"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(DEVICE == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(DEVICE == "cuda"),
    )

    all_results = []

    for model_name in ["mobilenet_v2", "mobilenet_v3_small"]:
        model_results = benchmark_single_model(
            model_name=model_name,
            train_loader=train_loader,
            test_loader=test_loader,
            output_dir=args.output_dir,
            epoch_options=EPOCH_OPTIONS,
            lr=args.lr,
        )
        all_results.extend(model_results)

    results_csv_path = args.output_dir / "benchmark_results.csv"
    results_json_path = args.output_dir / "benchmark_results.json"
    conclusion_path = args.output_dir / "benchmark_conclusion.txt"

    write_benchmark_csv(all_results, results_csv_path)

    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    conclusion_text = make_conclusion(all_results)
    print("\n" + conclusion_text)

    with open(conclusion_path, "w") as f:
        f.write(conclusion_text)

    print(f"\nSaved CSV        -> {results_csv_path}")
    print(f"Saved JSON       -> {results_json_path}")
    print(f"Saved conclusion -> {conclusion_path}")


if __name__ == "__main__":
    main()