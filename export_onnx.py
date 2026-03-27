"""
Step 3 — Export the fine-tuned model to ONNX for CPU inference on the server.

Exports the full classifier (features + classifier head + softmax) so the
server can use softmax probabilities directly as confidence scores.

Output names:
  "probs"  — softmax probabilities, shape (1, n_classes)

Usage:
    .venv/bin/python3 export_onnx.py

Output: data/card_embeddings.onnx  (commit this to the repo)
"""

import json
import os

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

WEIGHTS_PATH   = os.path.join("data", "finetuned_mobilenet.pth")
CLASS_MAP_PATH = os.path.join("data", "class_to_card_id.json")
ONNX_PATH      = os.path.join("data", "card_embeddings.onnx")

# Load class count from the saved map
with open(CLASS_MAP_PATH) as f:
    class_to_card_id = json.load(f)
n_classes = len(class_to_card_id)
print(f"Classes: {n_classes}")

# Reconstruct model and load weights
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model   = mobilenet_v3_small(weights=weights)
model.classifier[3] = nn.Linear(1024, n_classes)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
model.eval()

# Wrap with softmax so output is probabilities
classifier = nn.Sequential(model, nn.Softmax(dim=1))
classifier.eval()

dummy = torch.zeros(1, 3, 224, 224)
torch.onnx.export(
    classifier,
    dummy,
    ONNX_PATH,
    input_names=["image"],
    output_names=["probs"],
    dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}},
    opset_version=17,
)
print(f"Exported → {ONNX_PATH}  ({os.path.getsize(ONNX_PATH) / 1e6:.1f} MB)")
print("Next: run build_database.py to build the index, then commit the ONNX file.")
