"""
Phase 2 — Build the card embedding database.

Steps:
  1. Fetch full card list from YGOPRODECK API → data/cards.json
  2. Download small card images          → data/images/{card_id}.jpg
  3. Compute CNN embeddings for every image  → data/cnn_embeddings.npy
                                              data/cnn_card_ids.npy
                                              data/cnn_card_names.json
"""

import json
import os
import random
import time

import cv2
import numpy as np
import requests
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CARDS_JSON = os.path.join(DATA_DIR, "cards.json")
FAILED_TXT   = os.path.join(DATA_DIR, "failed_downloads.txt")
CNN_EMB_PATH = os.path.join(DATA_DIR, "cnn_embeddings.npy")
CNN_IDS_PATH = os.path.join(DATA_DIR, "cnn_card_ids.npy")
CNN_NAM_PATH = os.path.join(DATA_DIR, "cnn_card_names.json")

TCG_DATE_CUTOFF = "2010-04-26"  # only index cards released on or before this date (None = all cards)
SLEEP_MIN      = 0.3  # seconds between requests (min)
SLEEP_MAX      = 0.6  # seconds between requests (max) — jitter avoids pattern detection
BATCH_PAUSE_S  = 5.0  # longer pause every N requests
BATCH_SIZE     = 100  # how many requests between long pauses

# ---------------------------------------------------------------------------
# Step 1a — Fetch allowed card names (filtered by TCG release date)
# ---------------------------------------------------------------------------

SETS_JSON = os.path.join(DATA_DIR, "sets.json")

def fetch_allowed_card_names():
    """Return a set of card names whose sets were released on or before TCG_DATE_CUTOFF.
    Returns None if no cutoff is configured (include everything)."""
    if not TCG_DATE_CUTOFF:
        return None

    if os.path.exists(SETS_JSON):
        print(f"[1/4] sets.json already exists, skipping fetch.")
        with open(SETS_JSON) as f:
            all_sets = json.load(f)
    else:
        print("[1/4] Fetching set list from YGOPRODECK …")
        resp = requests.get("https://db.ygoprodeck.com/api/v7/cardsets.php", timeout=30)
        resp.raise_for_status()
        all_sets = resp.json()
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(SETS_JSON, "w") as f:
            json.dump(all_sets, f)

    allowed_set_names = {
        s["set_name"] for s in all_sets
        if s.get("tcg_date") and s["tcg_date"] <= TCG_DATE_CUTOFF
    }
    print(f"    {len(allowed_set_names)} sets released on or before {TCG_DATE_CUTOFF}")

    # Load cards.json to find which card names appear in those sets
    with open(CARDS_JSON) as f:
        cards = json.load(f)

    allowed_names = {
        card["name"] for card in cards
        if any(cs["set_name"] in allowed_set_names for cs in card.get("card_sets", []))
    }
    print(f"    {len(allowed_names):,} cards within cutoff")
    return allowed_names


# ---------------------------------------------------------------------------
# Step 1 — Fetch card list
# ---------------------------------------------------------------------------

def fetch_card_list():
    if os.path.exists(CARDS_JSON):
        print(f"[1/4] cards.json already exists, skipping fetch.")
        with open(CARDS_JSON) as f:
            return json.load(f)

    print("[1/4] Fetching card list from YGOPRODECK …")
    url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    cards = resp.json()["data"]
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CARDS_JSON, "w") as f:
        json.dump(cards, f)
    print(f"    Saved {len(cards):,} cards to {CARDS_JSON}")
    return cards


# ---------------------------------------------------------------------------
# Step 2 — Download card images
# ---------------------------------------------------------------------------

def download_images(cards):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    failed = []
    n_downloaded = 0

    # Flatten: one entry per image (some cards have alternate art)
    images_to_fetch = []
    for card in cards:
        for img in card.get("card_images", []):
            images_to_fetch.append({
                "card_id":   img["id"],
                "card_name": card["name"],
                "url":       img["image_url_small"],
            })

    already = sum(
        1 for e in images_to_fetch
        if os.path.exists(os.path.join(IMAGES_DIR, f"{e['card_id']}.jpg"))
    )
    print(f"[2/4] Downloading images ({already:,} already present, "
          f"{len(images_to_fetch) - already:,} to fetch) …")

    for entry in tqdm(images_to_fetch, unit="img"):
        path = os.path.join(IMAGES_DIR, f"{entry['card_id']}.jpg")
        if os.path.exists(path):
            continue
        try:
            r = requests.get(entry["url"], timeout=15)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            n_downloaded += 1
            # jittered sleep between every request
            time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
            # longer pause every BATCH_SIZE downloads
            if n_downloaded % BATCH_SIZE == 0:
                tqdm.write(f"    [{n_downloaded} downloaded] pausing {BATCH_PAUSE_S}s …")
                time.sleep(BATCH_PAUSE_S)
        except Exception as exc:
            failed.append(f"{entry['card_id']}\t{entry['card_name']}\t{exc}")

    if failed:
        with open(FAILED_TXT, "w") as f:
            f.write("\n".join(failed))
        print(f"    {len(failed)} failed downloads logged to {FAILED_TXT}")
    else:
        print("    All images downloaded successfully.")

    return images_to_fetch


# ---------------------------------------------------------------------------
# Step 3 — Compute CNN embeddings and export index
# ---------------------------------------------------------------------------

_transform = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224), antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def build_cnn_index(images_meta):
    if os.path.exists(CNN_EMB_PATH):
        print("[3/3] CNN index already exists, skipping.")
        return

    print("[3/3] Computing CNN embeddings (MobileNetV3-Small) …")
    weights   = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    m         = mobilenet_v3_small(weights=weights)
    extractor = torch.nn.Sequential(m.features, m.avgpool, torch.nn.Flatten(1))
    extractor.eval()

    all_embs   = []
    all_ids    = []
    id_to_name = {}

    for entry in tqdm(images_meta, unit="card"):
        path = os.path.join(IMAGES_DIR, f"{entry['card_id']}.jpg")
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = _transform(rgb).unsqueeze(0)
        with torch.no_grad():
            vec = extractor(tensor).squeeze().numpy()  # (576,)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        all_embs.append(vec)
        all_ids.append(entry["card_id"])
        id_to_name[entry["card_id"]] = entry["card_name"]

    emb_matrix = np.vstack(all_embs).astype(np.float32)
    ids_array  = np.array(all_ids, dtype=np.int32)

    np.save(CNN_EMB_PATH, emb_matrix)
    np.save(CNN_IDS_PATH, ids_array)
    with open(CNN_NAM_PATH, "w") as f:
        json.dump({str(k): v for k, v in id_to_name.items()}, f)

    print(f"    {emb_matrix.shape[0]:,} embeddings ({emb_matrix.shape[1]}d) "
          f"for {len(id_to_name):,} cards → {CNN_EMB_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    allowed_names = fetch_allowed_card_names()
    cards         = fetch_card_list()
    image_meta    = download_images(cards)

    if allowed_names is not None:
        before = len(image_meta)
        image_meta = [e for e in image_meta if e["card_name"] in allowed_names]
        print(f"[filter] {before:,} → {len(image_meta):,} images after date filter ({TCG_DATE_CUTOFF})")

    build_cnn_index(image_meta)
    print("\nDone. Database is ready.")
