"""
Phase 3 — Card detector.

Finds Yu-Gi-Oh card regions in a camera frame, deskews them, and returns
cropped 200×290 px images ready for hashing.
"""

import cv2
import numpy as np

# Output size (preserves YGO card proportions: 59mm × 86mm ≈ 0.686)
CARD_W = 200
CARD_H = 290

# Detection thresholds
MIN_AREA       = 8000
MAX_AREA_RATIO = 0.80   # fraction of total frame area
ASPECT_MIN     = 0.3    # min(w,h)/max(w,h) — finger intrusion makes this unreliable
ASPECT_MAX     = 1.0    # nearly square is fine, de-rotate handles the rest
GLARE_THRESH   = 240    # mean pixel value above this → low confidence


def sort_corners(pts):
    """Sort 4 points: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def is_card_back(region_bgr):
    """Return True if the region looks like a card back (dark purple/black)."""
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    # Card back is a dark purple — hue ~130–160, low value
    mask = cv2.inRange(hsv, (120, 30, 0), (165, 255, 80))
    ratio = mask.sum() / 255 / mask.size
    return ratio > 0.25


def detect_cards(frame_bgr):
    """
    Detect Yu-Gi-Oh cards in a BGR frame.

    Returns a list of dicts:
        {
            "image":      np.ndarray (200×290, BGR),
            "corners":    np.ndarray (4×2, float32) in original frame coords,
            "low_confidence": bool  (True if glare detected)
        }
    """
    h, w = frame_bgr.shape[:2]
    frame_area = h * w

    # --- pre-processing ---
    gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    edges  = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges  = cv2.dilate(edges, kernel, iterations=2)

    # --- contour detection ---
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA_RATIO * frame_area:
            continue

        hull = cv2.convexHull(cnt)
        rect = cv2.minAreaRect(hull)
        (cx, cy), (rw, rh), angle = rect
        if rh == 0 or rw == 0:
            continue
        aspect = min(rw, rh) / max(rw, rh)  # always <= 1; card ≈ 0.686
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue

        # De-rotate: for portrait rect (rh>=rw), correct angle is -(90+angle)
        # For landscape rect (rw>rh), swap dims and use -angle
        if rh >= rw:
            rotation = -(90 + angle)
        else:
            rotation = -angle
            rw, rh = rh, rw  # swap so rh is the longer (height) side

        # Finger intrusion makes the rect nearly square — recompute width
        # from the true card aspect ratio using the longer side as height
        CARD_ASPECT = 59.0 / 86.0  # ≈ 0.686
        crop_h = rh
        crop_w = rh * CARD_ASPECT

        M_rot   = cv2.getRotationMatrix2D((cx, cy), rotation, 1.0)
        rotated = cv2.warpAffine(frame_bgr, M_rot, (w, h))
        x1 = max(0, int(cx - crop_w / 2))
        y1 = max(0, int(cy - crop_h / 2))
        x2 = min(w, int(cx + crop_w / 2))
        y2 = min(h, int(cy + crop_h / 2))
        cropped = rotated[y1:y2, x1:x2]
        if cropped.size == 0:
            continue
        warped = cv2.resize(cropped, (CARD_W, CARD_H))

        # corners for drawing bounding box on the original frame
        box    = cv2.boxPoints(rect)
        approx = np.int32(box)
        corners = sort_corners(approx)

        if is_card_back(warped):
            continue

        mean_val     = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).mean()
        low_conf     = mean_val > GLARE_THRESH

        results.append({
            "image":          warped,
            "corners":        corners,
            "low_confidence": low_conf,
        })

    return results
