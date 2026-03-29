"""
Card recognition engine.

Uses the fine-tuned CNN (ONNX) if data/card_embeddings.onnx exists,
otherwise falls back to ORB + FLANN-LSH.
"""

import json
import os
from typing import Optional

import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# CNN paths
ONNX_PATH    = os.path.join(DATA_DIR, "card_embeddings.onnx")
CNN_EMB_PATH = os.path.join(DATA_DIR, "cnn_embeddings.npy")
CNN_IDS_PATH = os.path.join(DATA_DIR, "cnn_card_ids.npy")
CNN_NAM_PATH = os.path.join(DATA_DIR, "cnn_card_names.json")

# ORB paths
ORB_DES_PATH = os.path.join(DATA_DIR, "orb_descriptors.npy")
ORB_IDS_PATH = os.path.join(DATA_DIR, "orb_card_ids.npy")
ORB_NAM_PATH = os.path.join(DATA_DIR, "orb_card_names.json")

ORB_FEATURES     = 500
RATIO_THRESH     = 0.75
MIN_GOOD_MATCHES = 8
CNN_THRESHOLD    = 0.65  # cosine similarity threshold for embedding matches

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_mode       = None   # "cnn" or "orb"
_id_to_name = None

# CNN
_sess       = None
_embeddings = None
_card_ids   = None
_transform  = None

# ORB
_orb        = None
_flann      = None
_orb_ids    = None
_clahe      = None


def load_index():
    global _mode, _id_to_name

    if os.path.exists(ONNX_PATH) and os.path.exists(CNN_EMB_PATH):
        _load_cnn()
    else:
        _load_orb()


def _load_cnn():
    global _mode, _sess, _embeddings, _card_ids, _id_to_name, _transform
    import onnxruntime as ort
    from torchvision import transforms

    print("[recogniser] Loading fine-tuned CNN (ONNX) …", flush=True)
    _sess       = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    _embeddings = np.load(CNN_EMB_PATH)
    _card_ids   = np.load(CNN_IDS_PATH)

    with open(CNN_NAM_PATH) as f:
        _id_to_name = json.load(f)

    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    _mode = "cnn"
    print(f"[recogniser] CNN ready — {len(_card_ids):,} card embeddings.", flush=True)


def _load_orb():
    global _mode, _orb, _flann, _orb_ids, _id_to_name, _clahe

    print("[recogniser] Loading ORB descriptors …", flush=True)
    _orb   = cv2.ORB_create(nfeatures=ORB_FEATURES)
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    des_matrix = np.load(ORB_DES_PATH)
    _orb_ids   = np.load(ORB_IDS_PATH)

    with open(ORB_NAM_PATH) as f:
        raw = json.load(f)
    _id_to_name = {int(k): v for k, v in raw.items()}

    print(f"[recogniser] Building FLANN index over {des_matrix.shape[0]:,} descriptors …", flush=True)
    index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
    _flann = cv2.FlannBasedMatcher(index_params, dict(checks=50))
    _flann.add([des_matrix])
    _flann.train()

    _mode = "orb"
    print(f"[recogniser] ORB ready — {len(_id_to_name):,} cards indexed.", flush=True)


def get_valid_card_names() -> list:
    if _id_to_name is None:
        return []
    if _mode == "cnn":
        return sorted(set(_id_to_name.values()))
    return sorted(set(_id_to_name.values()))


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

def recognise_card(card_image_bgr: np.ndarray) -> Optional[dict]:
    if _mode == "cnn":
        return _recognise_cnn(card_image_bgr)
    return _recognise_orb(card_image_bgr)


def _recognise_cnn(card_bgr: np.ndarray) -> Optional[dict]:
    from PIL import Image as PILImage

    pil  = PILImage.fromarray(cv2.cvtColor(card_bgr, cv2.COLOR_BGR2RGB))
    t    = _transform(pil).unsqueeze(0).numpy()
    emb  = _sess.run(["embedding"], {"image": t})[0].squeeze()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    sims  = _embeddings @ emb
    idx   = int(np.argmax(sims))
    score = float(sims[idx])

    card_id = int(_card_ids[idx])
    name    = _id_to_name.get(str(card_id), str(card_id))
    print(f"[recognise] cnn score={score:.3f} ({name})", flush=True)

    if score < CNN_THRESHOLD:
        return None

    return {
        "card_id":    card_id,
        "card_name":  name,
        "confidence": round(score, 3),
    }


def _recognise_orb(card_bgr: np.ndarray) -> Optional[dict]:
    gray     = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = _clahe.apply(gray)
    _, des   = _orb.detectAndCompute(enhanced, None)
    if des is None or len(des) < 2:
        return None

    try:
        knn = _flann.knnMatch(des, k=2)
    except cv2.error:
        return None

    votes: dict = {}
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO_THRESH * n.distance:
            card_id = int(_orb_ids[m.trainIdx])
            votes[card_id] = votes.get(card_id, 0) + 1

    if not votes:
        return None

    best_id    = max(votes, key=votes.get)
    best_votes = votes[best_id]
    name       = _id_to_name.get(best_id, str(best_id))
    print(f"[recognise] orb votes={best_votes} ({name})", flush=True)

    if best_votes < MIN_GOOD_MATCHES:
        return None

    return {
        "card_id":      best_id,
        "card_name":    name,
        "confidence":   round(min(best_votes / 40, 1.0), 3),
        "good_matches": best_votes,
    }
