"""
Card recognition engine — MobileNetV3-Small CNN embeddings.

Loads pre-computed 576-dim feature vectors for all cards, then matches
query card images by cosine similarity (fast numpy dot product).
"""

import json
import os
from typing import Optional

# Set torch cache dir before importing torch so the model weights land in the
# project directory instead of $HOME/.cache (which may not be writable).
os.environ.setdefault(
    "TORCH_HOME",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".torch"),
)

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EMB_PATH  = os.path.join(DATA_DIR, "cnn_embeddings.npy")
IDS_PATH  = os.path.join(DATA_DIR, "cnn_card_ids.npy")
NAM_PATH  = os.path.join(DATA_DIR, "cnn_card_names.json")

SIMILARITY_THRESHOLD = 0.72   # cosine similarity; tune up if too many false matches

# ---------------------------------------------------------------------------
# Index — loaded once at startup
# ---------------------------------------------------------------------------

_extractor  = None
_embeddings = None   # (N, 576) float32, L2-normalised rows
_card_ids   = None   # (N,) int32
_id_to_name = None   # str(card_id) → card_name

_transform = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224), antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def _build_extractor():
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    m = mobilenet_v3_small(weights=weights)
    extractor = torch.nn.Sequential(m.features, m.avgpool, torch.nn.Flatten(1))
    extractor.eval()
    return extractor


def load_index():
    global _extractor, _embeddings, _card_ids, _id_to_name

    print("[recogniser] Loading MobileNetV3-Small …", flush=True)
    _extractor = _build_extractor()

    _embeddings = np.load(EMB_PATH)   # (N, 576) float32
    _card_ids   = np.load(IDS_PATH)   # (N,) int32

    with open(NAM_PATH) as f:
        _id_to_name = json.load(f)    # str(id) → name

    print(f"[recogniser] Ready — {len(_card_ids):,} card embeddings loaded.", flush=True)


def get_valid_card_names() -> list[str]:
    """Return a sorted, deduplicated list of all card names in the index."""
    if _id_to_name is None:
        return []
    return sorted(set(_id_to_name.values()))


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

def _embed(card_bgr: np.ndarray) -> np.ndarray:
    """Return an L2-normalised 576-dim embedding for a BGR card image."""
    rgb    = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2RGB)
    tensor = _transform(rgb).unsqueeze(0)          # (1, 3, 224, 224)
    with torch.no_grad():
        vec = _extractor(tensor).squeeze().numpy() # (576,)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def recognise_card(card_image_bgr: np.ndarray) -> Optional[dict]:
    """
    Match a BGR card image against the embedding index.

    Returns a dict on confident match, None otherwise:
        {
            "card_id":    int,
            "card_name":  str,
            "confidence": float  (0–1)
        }
    """
    emb   = _embed(card_image_bgr)
    sims  = _embeddings @ emb          # cosine similarity, shape (N,)
    idx   = int(np.argmax(sims))
    score = float(sims[idx])

    card_id = int(_card_ids[idx])
    name    = _id_to_name.get(str(card_id), str(card_id))
    print(f"[recognise] best={score:.3f} ({name})", flush=True)

    if score < SIMILARITY_THRESHOLD:
        return None
    return {
        "card_id":    card_id,
        "card_name":  _id_to_name.get(str(card_id), str(card_id)),
        "confidence": round(score, 3),
    }
