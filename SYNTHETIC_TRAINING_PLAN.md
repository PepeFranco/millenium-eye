# Synthetic Training Plan — Fine-tuned CNN Recognition

## Goal

Replace the current ORB feature matching with a MobileNetV3 model fine-tuned
on synthetic photos of Yu-Gi-Oh cards. This mirrors what Konami's Neuron app
likely does: generate realistic "fake photos" from clean reference scans, train
a classifier on them, and use the resulting embeddings for recognition.

## Why This Should Work

ORB and pHash both fail because they compare a camera photo against a studio
scan — a large domain gap. Fine-tuning teaches the model what physical card
photos look like directly, even if all training images are synthetic. Even
imperfect glare/lighting simulation is enough for the model to learn
robustness to real-world conditions.

## Prerequisites

- Full-size card images already downloaded in `data/images_full/` (421×614 px)
  — these are the training source. If not yet present, run `build_database.py`
  first (takes ~1 hour to download 3,817 images).
- PyTorch with MPS support (standard `pip install torch torchvision` on M1 Mac,
  MPS works out of the box since PyTorch 1.12)
- `pip install onnx onnxruntime` for export and server inference

## Step 1 — Generate Synthetic Training Data

Write `generate_training_data.py`. For each card in `data/images_full/`,
generate N_AUGMENTS (suggested: 50) variants and save to
`data/synthetic/{card_id}/{i}.jpg`.

Each variant applies a random combination of:

```python
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

def augment(img_bgr):
    h, w = img_bgr.shape[:2]

    # 1. Perspective distortion — simulates non-flat hold angle
    margin = 0.08
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = src + np.random.uniform(-margin*w, margin*w, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 2. Brightness + contrast
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(0.5, 1.5))
    img_pil = ImageEnhance.Contrast(img_pil).enhance(random.uniform(0.7, 1.4))

    # 3. Color temperature shift (warm/cool)
    img_np = np.array(img_pil).astype(np.float32)
    shift = random.uniform(-30, 30)
    img_np[:,:,0] = np.clip(img_np[:,:,0] + shift, 0, 255)   # R
    img_np[:,:,2] = np.clip(img_np[:,:,2] - shift, 0, 255)   # B

    # 4. Glare — bright ellipses simulating holographic reflection
    if random.random() < 0.5:
        n_glare = random.randint(1, 3)
        for _ in range(n_glare):
            cx, cy = random.randint(0, w), random.randint(0, h)
            rx, ry = random.randint(20, w//3), random.randint(10, h//4)
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask, (cx, cy), (rx, ry), random.uniform(0, 180),
                        0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            intensity = random.uniform(0.3, 0.8)
            img_np = np.clip(img_np + mask[:,:,None] * 255 * intensity, 0, 255)

    # 5. Gaussian noise
    noise = np.random.normal(0, random.uniform(2, 12), img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)

    # 6. JPEG re-compression (simulates camera compression)
    quality = random.randint(70, 95)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), encode_param)
    img_final = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return img_final
```

Save each augmented image resized to 224×224 (the MobileNetV3 input size).

Estimated time: ~1-2 hours on CPU for 3,817 cards × 50 = 190,850 images.
Estimated disk: ~2-3 GB.

## Step 2 — Train the Model

Write `train_model.py`.

```python
import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

SYNTHETIC_DIR = "data/synthetic"   # subdirs named by card_id
EPOCHS        = 10
BATCH_SIZE    = 64
LR            = 1e-4
DEVICE        = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Dataset: torchvision ImageFolder expects subdirs named by class
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
dataset    = datasets.ImageFolder(SYNTHETIC_DIR, transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)

# Load pretrained MobileNetV3, replace classifier head
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model   = mobilenet_v3_small(weights=weights)
model.classifier[3] = torch.nn.Linear(1024, len(dataset.classes))
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(dataloader):.4f}  "
          f"acc={correct/total:.3f}")

torch.save(model.state_dict(), "data/finetuned_mobilenet.pth")
# Also save class→card_id mapping from dataset.classes
```

Estimated time: 1-2 hours on M1 Pro MPS.

## Step 3 — Export to ONNX

Write `export_onnx.py`:

```python
import torch
import torchvision
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import json

N_CLASSES = 3817   # update to match actual class count

weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model   = mobilenet_v3_small(weights=weights)
model.classifier[3] = torch.nn.Linear(1024, N_CLASSES)
model.load_state_dict(torch.load("data/finetuned_mobilenet.pth", map_location="cpu"))

# Strip classifier — export embedding extractor only
extractor = torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten(1))
extractor.eval()

dummy = torch.zeros(1, 3, 224, 224)
torch.onnx.export(
    extractor, dummy, "data/card_embeddings.onnx",
    input_names=["image"], output_names=["embedding"],
    dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
    opset_version=17,
)
print("Exported data/card_embeddings.onnx")
```

Commit `data/card_embeddings.onnx` to the repo (~10 MB). The server never needs
PyTorch — only `onnxruntime` which is lightweight.

## Step 4 — Build Embedding Index

Add `build_cnn_index()` to `build_database.py` using `onnxruntime`:

```python
import onnxruntime as ort

def build_cnn_index(images_meta):
    sess = ort.InferenceSession("data/card_embeddings.onnx")
    # ... same loop as before, run sess.run(["embedding"], {"image": tensor})
    # save to data/cnn_embeddings.npy, data/cnn_card_ids.npy, data/cnn_card_names.json
```

## Step 5 — Update recogniser.py

Same cosine similarity approach as the previous CNN attempt, but using
`onnxruntime` for inference instead of PyTorch:

```python
import onnxruntime as ort

sess = ort.InferenceSession("data/card_embeddings.onnx")

def _embed(card_bgr):
    # preprocess → (1, 3, 224, 224) float32
    emb = sess.run(["embedding"], {"image": tensor})[0].squeeze()
    return emb / np.linalg.norm(emb)
```

Threshold: start at 0.75. Because the model was trained on synthetic card
photos (not ImageNet classes), embeddings will be much more discriminative
and cosine similarity will be meaningful.

## Files to Create

- `generate_training_data.py` — augmentation pipeline
- `train_model.py` — fine-tuning script (run on M1 Mac, not server)
- `export_onnx.py` — export trained model to ONNX
- Update `build_database.py` — add `build_cnn_index()` using onnxruntime
- Update `recogniser.py` — use onnxruntime for inference
- Add `onnxruntime` to `requirements.txt`

## Notes

- Training must be run on the M1 Mac (or any machine with a GPU/MPS).
  The server only needs `onnxruntime` for inference.
- The ONNX model file (~10 MB) should be committed to the repo so the server
  gets it via `git pull`.
- `data/synthetic/` (~2-3 GB) should NOT be committed — add to `.gitignore`.
- If accuracy is still insufficient after 50 augments/card, increase
  N_AUGMENTS to 100-200 and re-train. More variety = better robustness.
- The class→card_id mapping from `dataset.classes` must be saved alongside
  the model so the embedding index can be built correctly.
