"""
Step 1 — Generate synthetic training data.

For each full-size card image in data/images_full/, generate N_AUGMENTS
augmented variants simulating real-world photo conditions:
  - Binder scene: target card surrounded by partial edges of other cards
  - Aggressive perspective distortion (card held at steep angles)
  - Partial crop (card not fully in frame — edges cut off)
  - Brightness extremes and uneven lighting gradients
  - Heavy glare that wipes out entire card regions
  - Motion blur (shaky hands)
  - Sleeve and finger occlusion
  - Gaussian noise + JPEG recompression

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

IMAGES_DIR    = os.path.join("data", "images_full")
SYNTHETIC_DIR = os.path.join("data", "synthetic")
N_AUGMENTS    = 80
OUTPUT_SIZE   = 224


def make_binder_scene(img_bgr, all_image_paths):
    """
    Place the target card in a canvas with partial edges of neighbouring cards
    visible — simulates a binder where adjacent cards bleed into frame.
    Returns a BGR image the same size as img_bgr.
    """
    h, w = img_bgr.shape[:2]

    # How much of neighbouring cards to show (10–35% of card dimension)
    pad_x = int(w * random.uniform(0.10, 0.35))
    pad_y = int(h * random.uniform(0.10, 0.35))

    canvas_h = h + 2 * pad_y
    canvas_w = w + 2 * pad_x
    canvas   = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Fill each padded edge with a slice of a random neighbour card
    sides = random.sample(["left", "right", "top", "bottom"],
                          random.randint(1, 4))
    for side in sides:
        path = random.choice(all_image_paths)
        nbr  = cv2.imread(path)
        if nbr is None:
            continue
        nbr = cv2.resize(nbr, (w, h))
        if side == "left":
            canvas[pad_y:pad_y + h, :pad_x] = nbr[:, w - pad_x:]
        elif side == "right":
            canvas[pad_y:pad_y + h, pad_x + w:] = nbr[:, :pad_x]
        elif side == "top":
            canvas[:pad_y, pad_x:pad_x + w] = nbr[h - pad_y:, :]
        else:  # bottom
            canvas[pad_y + h:, pad_x:pad_x + w] = nbr[:pad_y, :]

    # Place target card in centre
    canvas[pad_y:pad_y + h, pad_x:pad_x + w] = img_bgr

    # Resize the whole scene back to the original card size
    return cv2.resize(canvas, (w, h))


def augment(img_bgr, all_image_paths):
    h, w = img_bgr.shape[:2]

    # 0. Binder scene — 50% of the time
    if random.random() < 0.5:
        img_bgr = make_binder_scene(img_bgr, all_image_paths)

    img_np = img_bgr.astype(np.float32)

    # 1. Aggressive perspective distortion
    margin = random.uniform(0.05, 0.22)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.random.uniform(-margin * w, margin * w, src.shape).astype(np.float32)
    M   = cv2.getPerspectiveTransform(src, dst)
    img_np = cv2.warpPerspective(img_np, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 2. Partial crop — simulate card not fully in frame
    if random.random() < 0.5:
        crop_side = random.choice(["top", "bottom", "left", "right"])
        frac = random.uniform(0.05, 0.25)
        ch, cw = img_np.shape[:2]
        if crop_side == "top":
            cut = int(ch * frac)
            img_np = np.pad(img_np[cut:], ((cut, 0), (0, 0), (0, 0)), mode="edge")
        elif crop_side == "bottom":
            cut = int(ch * frac)
            img_np = np.pad(img_np[:-cut], ((0, cut), (0, 0), (0, 0)), mode="edge")
        elif crop_side == "left":
            cut = int(cw * frac)
            img_np = np.pad(img_np[:, cut:], ((0, 0), (cut, 0), (0, 0)), mode="edge")
        else:
            cut = int(cw * frac)
            img_np = np.pad(img_np[:, :-cut], ((0, 0), (0, cut), (0, 0)), mode="edge")

    # 3. Uneven lighting gradient
    if random.random() < 0.5:
        ch, cw = img_np.shape[:2]
        direction   = random.choice(["h", "v"])
        bright_side = random.uniform(0.6, 1.5)
        dark_side   = random.uniform(0.3, 0.9)
        if direction == "h":
            grad = np.linspace(bright_side, dark_side, cw, dtype=np.float32)
            grad = np.tile(grad, (ch, 1))
        else:
            grad = np.linspace(bright_side, dark_side, ch, dtype=np.float32)
            grad = np.tile(grad[:, None], (1, cw))
        img_np = np.clip(img_np * grad[:, :, None], 0, 255)

    # 4. Brightness + contrast (via PIL — converts BGR→RGB internally)
    img_pil = Image.fromarray(cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_BGR2RGB))
    img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(0.3, 1.8))
    img_pil = ImageEnhance.Contrast(img_pil).enhance(random.uniform(0.5, 1.6))
    img_np  = np.array(img_pil).astype(np.float32)  # now RGB

    # 5. Color temperature shift
    shift = random.uniform(-45, 45)
    img_np[:, :, 0] = np.clip(img_np[:, :, 0] + shift, 0, 255)   # R
    img_np[:, :, 2] = np.clip(img_np[:, :, 2] - shift, 0, 255)   # B

    # 6. Heavy glare
    ch, cw = img_np.shape[:2]
    if random.random() < 0.75:
        for _ in range(random.randint(1, 4)):
            cx        = random.randint(-cw // 4, cw + cw // 4)
            cy        = random.randint(-ch // 4, ch + ch // 4)
            rx        = random.randint(cw // 6, cw)
            ry        = random.randint(ch // 8, ch // 2)
            ang       = random.uniform(0, 180)
            intensity = random.uniform(0.3, 1.0)
            mask = np.zeros((ch, cw), dtype=np.float32)
            cv2.ellipse(mask, (cx, cy), (rx, ry), ang, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (75, 75), 0)
            img_np = np.clip(img_np + mask[:, :, None] * 255 * intensity, 0, 255)

    # 7. Finger / object occlusion
    if random.random() < 0.3:
        ch, cw = img_np.shape[:2]
        side  = random.choice(["top", "bottom", "left", "right"])
        frac  = random.uniform(0.05, 0.20)
        color = np.array([random.randint(20, 100)] * 3, dtype=np.float32)
        if side == "top":
            img_np[:int(ch * frac)] = color
        elif side == "bottom":
            img_np[-int(ch * frac):] = color
        elif side == "left":
            img_np[:, :int(cw * frac)] = color
        else:
            img_np[:, -int(cw * frac):] = color

    # 8. Sleeve simulation
    if random.random() < 0.35:
        overlay = np.full_like(img_np, [210, 220, 240], dtype=np.float32)
        alpha   = random.uniform(0.05, 0.20)
        img_np  = np.clip(img_np * (1 - alpha) + overlay * alpha, 0, 255)

    # 9. Motion blur
    if random.random() < 0.4:
        ksize = random.choice([3, 5, 7, 9])
        angle = random.uniform(0, 180)
        k = np.zeros((ksize, ksize), dtype=np.float32)
        k[ksize // 2, :] = 1.0 / ksize
        rot = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle, 1)
        k   = cv2.warpAffine(k, rot, (ksize, ksize))
        img_np = cv2.filter2D(img_np, -1, k)

    # 10. Gaussian noise
    sigma  = random.uniform(3, 20)
    img_np = np.clip(img_np + np.random.normal(0, sigma, img_np.shape), 0, 255).astype(np.uint8)

    # 11. JPEG re-compression (RGB→BGR before encoding)
    quality      = random.randint(55, 95)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf       = cv2.imencode(".jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), encode_param)
    img_final    = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return cv2.resize(img_final, (OUTPUT_SIZE, OUTPUT_SIZE))


def main():
    import json
    with open(os.path.join("data", "orb_card_names.json")) as f:
        allowed_ids = set(json.load(f).keys())

    all_files   = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]
    image_files = [f for f in all_files if f.replace(".jpg", "") in allowed_ids]
    all_image_paths = [os.path.join(IMAGES_DIR, f) for f in image_files]

    print(f"Total images: {len(all_files):,}  →  Edison-legal: {len(image_files):,}")
    print(f"Generating {N_AUGMENTS} variants each → {len(image_files) * N_AUGMENTS:,} total")
    print(f"Output: {SYNTHETIC_DIR}/")

    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    skipped = 0

    for fname in tqdm(image_files, unit="card"):
        card_id = fname.replace(".jpg", "")
        out_dir = os.path.join(SYNTHETIC_DIR, card_id)

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
            aug = augment(img, all_image_paths)
            cv2.imwrite(out_path, aug, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    print(f"\nDone. ({skipped} cards already had full augmentation, skipped.)")
    print(f"Synthetic data in: {SYNTHETIC_DIR}/")


if __name__ == "__main__":
    main()
