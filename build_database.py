"""
Phase 2 — Build the card hash database.

Steps:
  1. Fetch full card list from YGOPRODECK API → data/cards.json
  2. Download small card images          → data/images/{card_id}.jpg
  3. Compute ORB descriptors for every image → data/orb_descriptors.npy
                                              data/orb_card_ids.npy
                                              data/orb_card_names.json
"""

import json
import os
import pickle
import random
import sqlite3
import time

import cv2
import numpy as np
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CARDS_JSON = os.path.join(DATA_DIR, "cards.json")
FAILED_TXT   = os.path.join(DATA_DIR, "failed_downloads.txt")
ORB_DES_PATH = os.path.join(DATA_DIR, "orb_descriptors.npy")
ORB_IDS_PATH = os.path.join(DATA_DIR, "orb_card_ids.npy")
ORB_NAM_PATH = os.path.join(DATA_DIR, "orb_card_names.json")

ORB_FEATURES   = 200  # keypoints per card image
SLEEP_MIN      = 0.3  # seconds between requests (min)
SLEEP_MAX      = 0.6  # seconds between requests (max) — jitter avoids pattern detection
BATCH_PAUSE_S  = 5.0  # longer pause every N requests
BATCH_SIZE     = 100  # how many requests between long pauses

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
# Step 3 — Compute ORB descriptors and export index
# ---------------------------------------------------------------------------

def build_orb_index(images_meta):
    if os.path.exists(ORB_DES_PATH):
        print("[3/3] ORB index already exists, skipping.")
        return

    print("[3/3] Computing ORB descriptors …")
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

    all_des   = []   # list of (N, 32) uint8 arrays
    all_ids   = []   # card_id repeated N times per image
    id_to_name = {}  # card_id → card_name

    for entry in tqdm(images_meta, unit="card"):
        path = os.path.join(IMAGES_DIR, f"{entry['card_id']}.jpg")
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        _, des = orb.detectAndCompute(img, None)
        if des is None:
            continue
        all_des.append(des)
        all_ids.extend([entry["card_id"]] * len(des))
        id_to_name[entry["card_id"]] = entry["card_name"]

    des_matrix = np.vstack(all_des).astype(np.uint8)
    ids_array  = np.array(all_ids, dtype=np.int32)

    np.save(ORB_DES_PATH, des_matrix)
    np.save(ORB_IDS_PATH, ids_array)
    with open(ORB_NAM_PATH, "w") as f:
        json.dump({str(k): v for k, v in id_to_name.items()}, f)

    print(f"    {des_matrix.shape[0]:,} descriptors for {len(id_to_name):,} cards → {ORB_DES_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cards      = fetch_card_list()
    image_meta = download_images(cards)
    build_orb_index(image_meta)
    print("\nDone. Database is ready.")
