"""
Card recognition engine — perceptual hashing on the artwork region.

Crops the artwork area from each card (top 55%, inset from the border),
computes a 64-bit pHash, and matches by minimum Hamming distance.
"""

import json
import os
from typing import Optional

import cv2
import imagehash
import numpy as np
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EMB_PATH  = os.path.join(DATA_DIR, "cnn_embeddings.npy")   # kept for compat check
IDS_PATH  = os.path.join(DATA_DIR, "phash_card_ids.npy")
NAM_PATH  = os.path.join(DATA_DIR, "phash_card_names.json")
HSH_PATH  = os.path.join(DATA_DIR, "phash_hashes.npy")

# Artwork region as fractions of card dimensions (200×290 px warped output)
ART_TOP    = 0.13
ART_BOTTOM = 0.57
ART_LEFT   = 0.07
ART_RIGHT  = 0.93

HAMMING_THRESHOLD = 15   # out of 64 bits; lower = stricter

# ---------------------------------------------------------------------------
# Index — loaded once at startup
# ---------------------------------------------------------------------------

_hashes     = None   # (N, 8) uint8 — 64-bit pHash packed into bytes
_card_ids   = None   # (N,) int32
_id_to_name = None   # str(card_id) → card_name


def _art_crop(bgr):
    """Return PIL Image of the artwork region of a 200×290 card image."""
    h, w = bgr.shape[:2]
    y1 = int(h * ART_TOP);    y2 = int(h * ART_BOTTOM)
    x1 = int(w * ART_LEFT);   x2 = int(w * ART_RIGHT)
    return Image.fromarray(cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))


def _phash_bits(pil_img) -> np.ndarray:
    """Return a 64-bit pHash as an (8,) uint8 array."""
    h = imagehash.phash(pil_img, hash_size=8)  # 8×8 = 64 bits
    return np.packbits(h.hash.flatten())        # (8,) uint8


def load_index():
    global _hashes, _card_ids, _id_to_name

    print("[recogniser] Loading pHash index …", flush=True)
    _hashes   = np.load(HSH_PATH)   # (N, 8) uint8
    _card_ids = np.load(IDS_PATH)   # (N,) int32

    with open(NAM_PATH) as f:
        _id_to_name = json.load(f)

    print(f"[recogniser] Ready — {len(_card_ids):,} card hashes loaded.", flush=True)


def get_valid_card_names() -> list:
    if _id_to_name is None:
        return []
    return sorted(set(_id_to_name.values()))


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

def _hamming_distances(query_bits: np.ndarray) -> np.ndarray:
    """Vectorised Hamming distance between query and all index hashes."""
    xor = _hashes ^ query_bits          # (N, 8) uint8
    # popcount each byte via lookup table
    bits = np.unpackbits(xor, axis=1)   # (N, 64) uint8 (0/1)
    return bits.sum(axis=1)             # (N,) int


def recognise_card(card_image_bgr: np.ndarray) -> Optional[dict]:
    """
    Match a BGR card image against the pHash index.

    Returns a dict on confident match, None otherwise:
        {
            "card_id":    int,
            "card_name":  str,
            "confidence": float  (0–1, higher = better)
        }
    """
    art   = _art_crop(card_image_bgr)
    bits  = _phash_bits(art)
    dists = _hamming_distances(bits)
    idx   = int(np.argmin(dists))
    dist  = int(dists[idx])

    card_id = int(_card_ids[idx])
    name    = _id_to_name.get(str(card_id), str(card_id))
    print(f"[recognise] best dist={dist} ({name})", flush=True)

    if dist > HAMMING_THRESHOLD:
        return None

    confidence = round(1.0 - dist / 64, 3)
    return {
        "card_id":    card_id,
        "card_name":  name,
        "confidence": confidence,
    }
