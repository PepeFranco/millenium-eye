"""
Step 2 — Fine-tune MobileNetV3-Small on synthetic card photos.

Reads data/synthetic/{card_id}/{i}.jpg, trains a classifier with one
class per card, saves the weights to data/finetuned_mobilenet.pth and
the class→card_id mapping to data/class_to_card_id.json.

Also merges any real images from data/training_samples/{card_id}/ —
only numeric subdirectories are used (incorrect/ and unrecognized/ are
skipped).

Run on M1 Mac (uses MPS GPU). Do NOT run on the server.

Usage:
    .venv/bin/python3 train_model.py
"""

import json
import os

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from tqdm import tqdm

SYNTHETIC_DIR  = os.path.join("data", "synthetic")
REAL_DIR       = os.path.join("data", "training_samples")
WEIGHTS_PATH   = os.path.join("data", "finetuned_mobilenet.pth")
CLASS_MAP_PATH = os.path.join("data", "class_to_card_id.json")


class _NumericImageFolder(datasets.ImageFolder):
    """ImageFolder that only loads from numeric (card_id) subdirectories."""
    def find_classes(self, directory):
        classes = sorted(
            d for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d)) and d.isdigit()
        )
        return classes, {cls: i for i, cls in enumerate(classes)}

EPOCHS      = 8
BATCH_SIZE  = 64
LR          = 3e-4
NUM_WORKERS = 4

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU (this will be slow)")

# ---------------------------------------------------------------------------
# Dataset transform (defined at module level — safe for multiprocessing)
# ---------------------------------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

if __name__ == "__main__":
    synthetic = datasets.ImageFolder(SYNTHETIC_DIR, transform=train_transform)

    # Merge real samples if the folder exists and has any numeric card dirs
    real_count = 0
    combined   = synthetic
    if os.path.isdir(REAL_DIR):
        real = _NumericImageFolder(REAL_DIR, transform=train_transform)
        if real.samples:
            # Remap real dataset's local class indices → synthetic class indices.
            # Classes not present in synthetic are silently dropped.
            remap = {
                local_idx: synthetic.class_to_idx[cls]
                for cls, local_idx in real.class_to_idx.items()
                if cls in synthetic.class_to_idx
            }
            real.samples = [(p, remap[t]) for p, t in real.samples if t in remap]
            real.targets  = [remap[t] for t in real.targets  if t in remap]
            real_count    = len(real.samples)
            combined      = ConcatDataset([synthetic, real])

    n_classes = len(synthetic.classes)
    print(f"Classes (cards): {n_classes:,}  |  Synthetic: {len(synthetic):,}  |  Real: {real_count:,}  |  Total: {len(combined):,}")

    dataloader = DataLoader(
        combined,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # Save class→card_id mapping (ImageFolder sorts dirs alphabetically)
    class_to_card_id = {i: int(cls) for i, cls in enumerate(synthetic.classes)}
    with open(CLASS_MAP_PATH, "w") as f:
        json.dump(class_to_card_id, f)
    print(f"Class map saved → {CLASS_MAP_PATH}")

    # ---------------------------------------------------------------------------
    # Model — pretrained MobileNetV3-Small, replace head for n_classes
    # ---------------------------------------------------------------------------

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model   = mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(1024, n_classes)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(dataloader), epochs=EPOCHS
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for imgs, labels in tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct    += (out.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch:>2}/{EPOCHS}  loss={total_loss/len(dataloader):.4f}  acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"  ✓ saved (best so far: {best_acc:.3f})")

    print(f"\nTraining complete. Best acc={best_acc:.3f}")
    print(f"Weights saved → {WEIGHTS_PATH}")
