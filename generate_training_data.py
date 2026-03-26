"""
Step 1 — Generate synthetic training data.

For each full-size card image in data/images_full/, generate N_AUGMENTS
augmented variants simulating real-world photo conditions (perspective
distortion, lighting variation, glare, noise, JPEG compression).

Output: data/synthetic/{card_id}/{i}.jpg  (224×224 px)

Usage:
    .venv/bin/python3 generate_training_data.py
"""

import os
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

IMAGES_DIR   = os.path.join("data", "images_full")
SYNTHETIC_DIR = os.path.join("data", "synthetic")
N_AUGMENTS   = 60    # augmented variants per card
OUTPUT_SIZE  = 224   # MobileNetV3 input size


def augment(img_bgr):
    h, w = img_bgr.shape[:2]

    # 1. Perspective distortion — simulates non-flat hold angle
    margin = random.uniform(0.02, 0.10)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.random.uniform(-margin * w, margin * w, src.shape).astype(np.float32)
    M   = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 2. Brightness + contrast
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(0.45, 1.6))
    img_pil = ImageEnhance.Contrast(img_pil).enhance(random.uniform(0.65, 1.45))

    # 3. Color temperature shift (warm / cool room lighting)
    img_np = np.array(img_pil).astype(np.float32)
    shift  = random.uniform(-35, 35)
    img_np[:, :, 0] = np.clip(img_np[:, :, 0] + shift, 0, 255)   # R
    img_np[:, :, 2] = np.clip(img_np[:, :, 2] - shift, 0, 255)   # B

    # 4. Glare — bright ellipses simulating holographic foil reflection
    if random.random() < 0.6:
        for _ in range(random.randint(1, 3)):
            cx  = random.randint(0, w)
            cy  = random.randint(0, h)
            rx  = random.randint(15, w // 3)
            ry  = random.randint(8,  h // 4)
            ang = random.uniform(0, 180)
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask, (cx, cy), (rx, ry), ang, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            img_np = np.clip(img_np + mask[:, :, None] * 255 * random.uniform(0.2, 0.7), 0, 255)

    # 5. Sleeve simulation — faint blue-tinted transparent overlay
    if random.random() < 0.35:
        overlay        = np.full_like(img_np, [210, 220, 240], dtype=np.float32)
        alpha          = random.uniform(0.05, 0.18)
        img_np         = np.clip(img_np * (1 - alpha) + overlay * alpha, 0, 255)

    # 6. Gaussian noise
    sigma  = random.uniform(2, 15)
    img_np = np.clip(img_np + np.random.normal(0, sigma, img_np.shape), 0, 255).astype(np.uint8)

    # 7. JPEG re-compression
    quality      = random.randint(65, 95)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf       = cv2.imencode(".jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), encode_param)
    img_final    = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return cv2.resize(img_final, (OUTPUT_SIZE, OUTPUT_SIZE))


def main():
    # Only train on Edison-legal cards (filtered by TCG date in build_database.py)
    import json
    with open(os.path.join("data", "orb_card_names.json")) as f:
        allowed_ids = set(json.load(f).keys())  # str card_ids

    all_files   = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]
    image_files = [f for f in all_files if f.replace(".jpg", "") in allowed_ids]
    print(f"Total images: {len(all_files):,}  →  Edison-legal: {len(image_files):,}")
    print(f"Generating {N_AUGMENTS} variants each → {len(image_files) * N_AUGMENTS:,} total")
    print(f"Output: {SYNTHETIC_DIR}/")

    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    skipped = 0

    for fname in tqdm(image_files, unit="card"):
        card_id  = fname.replace(".jpg", "")
        out_dir  = os.path.join(SYNTHETIC_DIR, card_id)

        # Skip if already fully generated
        if os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= N_AUGMENTS:
            skipped += 1
            continue

        img = cv2.imread(os.path.join(IMAGES_DIR, fname))
        if img is None:
            continue

        os.makedirs(out_dir, exist_ok=True)
        for i in range(N_AUGMENTS):
            out_path = os.path.join(out_dir, f"{i}.jpg")
            if os.path.exists(out_path):
                continue
            aug = augment(img)
            cv2.imwrite(out_path, aug, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    print(f"\nDone. ({skipped} cards already had full augmentation, skipped.)")
    print(f"Synthetic data in: {SYNTHETIC_DIR}/")


if __name__ == "__main__":
    main()
