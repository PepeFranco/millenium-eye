"""
Card recognition engine — ORB + FLANN-LSH.

Loads pre-computed ORB descriptors for all cards, builds a FLANN-LSH index,
and matches query card images by voting on nearest-neighbour descriptors.
"""

import json
import os
from typing import Optional

import cv2
import numpy as np

DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
ORB_DES_PATH = os.path.join(DATA_DIR, "orb_descriptors.npy")
ORB_IDS_PATH = os.path.join(DATA_DIR, "orb_card_ids.npy")
ORB_NAM_PATH = os.path.join(DATA_DIR, "orb_card_names.json")

ORB_FEATURES    = 500   # must match build_database.py
RATIO_THRESH    = 0.75  # Lowe ratio test
MIN_GOOD_MATCHES = 8    # minimum votes to accept a match

# ---------------------------------------------------------------------------
# Index — loaded once at startup
# ---------------------------------------------------------------------------

_orb        = None
_flann      = None
_card_ids   = None
_id_to_name = None
_clahe      = None


def load_index():
    global _orb, _flann, _card_ids, _id_to_name, _clahe

    _orb   = cv2.ORB_create(nfeatures=ORB_FEATURES)
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    print("[recogniser] Loading ORB descriptors …", flush=True)
    des_matrix = np.load(ORB_DES_PATH)
    _card_ids  = np.load(ORB_IDS_PATH)

    with open(ORB_NAM_PATH) as f:
        raw = json.load(f)
    _id_to_name = {int(k): v for k, v in raw.items()}

    print(f"[recogniser] Building FLANN index over {des_matrix.shape[0]:,} descriptors …", flush=True)
    index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
    _flann = cv2.FlannBasedMatcher(index_params, dict(checks=50))
    _flann.add([des_matrix])
    _flann.train()

    print(f"[recogniser] Ready — {len(_id_to_name):,} cards indexed.", flush=True)


def get_valid_card_names() -> list:
    if _id_to_name is None:
        return []
    return sorted(set(_id_to_name.values()))


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

def recognise_card(card_image_bgr: np.ndarray) -> Optional[dict]:
    """
    Match a BGR card image against the ORB index.

    Returns a dict on confident match, None otherwise:
        {
            "card_id":    int,
            "card_name":  str,
            "confidence": float,
            "good_matches": int
        }
    """
    gray     = cv2.cvtColor(card_image_bgr, cv2.COLOR_BGR2GRAY)
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
            card_id = int(_card_ids[m.trainIdx])
            votes[card_id] = votes.get(card_id, 0) + 1

    if not votes:
        return None

    best_id    = max(votes, key=votes.get)
    best_votes = votes[best_id]

    name = _id_to_name.get(best_id, str(best_id))
    print(f"[recognise] votes={best_votes} ({name})", flush=True)

    if best_votes < MIN_GOOD_MATCHES:
        return None

    return {
        "card_id":      best_id,
        "card_name":    name,
        "confidence":   round(min(best_votes / 40, 1.0), 3),
        "good_matches": best_votes,
    }
