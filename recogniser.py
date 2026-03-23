"""
Phase 4 — Card recognition engine (ORB + FLANN).

Loads pre-computed ORB descriptors for all cards, builds a FLANN-LSH index,
and matches query card images by voting on nearest-neighbour descriptors.
"""

import json
import logging
from typing import Optional

import cv2
import numpy as np

ORB_DES_PATH   = "data/orb_descriptors.npy"
ORB_IDS_PATH   = "data/orb_card_ids.npy"
ORB_NAM_PATH   = "data/orb_card_names.json"

ORB_FEATURES   = 200   # must match build_database.py
RATIO_THRESH   = 0.75  # Lowe ratio test
MIN_GOOD_MATCHES = 6   # minimum votes to accept a match

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Index — loaded once at startup
# ---------------------------------------------------------------------------

_orb        = None
_flann      = None
_card_ids   = None   # np.int32 array, one entry per descriptor row
_id_to_name = None   # dict int → str
_clahe      = None   # CLAHE instance for contrast normalisation


def load_index(des_path=ORB_DES_PATH, ids_path=ORB_IDS_PATH, nam_path=ORB_NAM_PATH):
    global _orb, _flann, _card_ids, _id_to_name, _clahe

    _orb   = cv2.ORB_create(nfeatures=ORB_FEATURES)
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    print("[recogniser] Loading ORB descriptors …")
    des_matrix = np.load(des_path)   # (N, 32) uint8
    _card_ids  = np.load(ids_path)   # (N,) int32

    with open(nam_path) as f:
        raw = json.load(f)
    _id_to_name = {int(k): v for k, v in raw.items()}

    print(f"[recogniser] Building FLANN index over {des_matrix.shape[0]:,} descriptors …")
    # LSH index for binary (ORB/BRIEF) descriptors
    index_params = dict(algorithm=6,          # FLANN_INDEX_LSH
                        table_number=12,
                        key_size=20,
                        multi_probe_level=2)
    _flann = cv2.FlannBasedMatcher(index_params, dict(checks=50))
    _flann.add([des_matrix])
    _flann.train()

    print(f"[recogniser] Ready — {len(_id_to_name):,} cards indexed.")


def _ensure_loaded():
    if _flann is None:
        load_index()


def get_valid_card_names() -> list[str]:
    """Return a sorted, deduplicated list of all card names in the index."""
    _ensure_loaded()
    return sorted(set(_id_to_name.values()))


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

def recognise_card(card_image_bgr: np.ndarray) -> Optional[dict]:
    """
    Match a BGR card image against the ORB index.

    Returns a result dict on confident match, None otherwise:
        {
            "card_id":      int,
            "card_name":    str,
            "confidence":   float,   # good_matches / total_query_descriptors
            "good_matches": int
        }
    """
    _ensure_loaded()

    gray = cv2.cvtColor(card_image_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = _clahe.apply(gray)
    _, des = _orb.detectAndCompute(enhanced, None)
    if des is None or len(des) < 2:
        return None

    # knnMatch with k=2 for Lowe ratio test
    try:
        knn = _flann.knnMatch(des, k=2)
    except cv2.error:
        return None

    # Collect good match votes
    votes: dict[int, int] = {}
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

    if best_votes < MIN_GOOD_MATCHES:
        return None

    return {
        "card_id":      best_id,
        "card_name":    _id_to_name.get(best_id, str(best_id)),
        "confidence":   round(min(best_votes / 30, 1.0), 4),  # 30 matches = 100%
        "good_matches": best_votes,
    }
