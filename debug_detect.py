"""
Debug script — runs card_detector on an image file and shows what it finds.

Usage:
    .venv/bin/python3 debug_detect.py path/to/image.jpg

Saves annotated debug images alongside the input file.
Does NOT require the ORB database.
"""

import sys
import os
import cv2
import numpy as np

from card_detector import (
    detect_cards, is_card_back,
    MIN_AREA, MAX_AREA_RATIO, ASPECT_MIN, ASPECT_MAX, GLARE_THRESH,
    CARD_W, CARD_H,
)


def main(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: could not load image: {image_path}")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Image: {w}x{h}")

    # ── replicate card_detector internals for verbose output ──────────────────
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    edges  = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges  = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Contours found: {len(contours)}")

    frame_area = h * w
    candidates = 0
    rejected   = {"area": 0, "aspect": 0, "card_back": 0, "glare": 0}

    debug_frame = frame.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA_RATIO * frame_area:
            rejected["area"] += 1
            continue

        hull = cv2.convexHull(cnt)
        rect = cv2.minAreaRect(hull)
        (cx, cy), (rw, rh), angle = rect
        if rh == 0 or rw == 0:
            continue
        aspect = min(rw, rh) / max(rw, rh)
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            rejected["aspect"] += 1
            continue

        candidates += 1

        # Draw this candidate on the debug frame
        box = cv2.boxPoints(rect)
        cv2.drawContours(debug_frame, [np.int32(box)], 0, (0, 255, 255), 2)
        cv2.putText(debug_frame, f"a={aspect:.2f}", (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    print(f"\nRejected by area filter:   {rejected['area']}")
    print(f"Rejected by aspect filter: {rejected['aspect']}")
    print(f"Candidates after filters:  {candidates}")

    # ── now run the full detector ─────────────────────────────────────────────
    results = detect_cards(frame)
    print(f"\nCards returned by detect_cards: {len(results)}")

    out_frame = frame.copy()
    for i, det in enumerate(results):
        corners = det["corners"]
        low_conf = det["low_confidence"]
        color = (0, 255, 0) if not low_conf else (0, 255, 255)
        cv2.polylines(out_frame, [corners.astype(np.int32)], True, color, 3)

        # Save the crop
        crop_path = image_path.replace(".", f"_crop{i}.")
        cv2.imwrite(crop_path, det["image"])
        print(f"  Card {i}: low_confidence={low_conf}  → saved crop: {crop_path}")

        # Check card back manually
        back = is_card_back(det["image"])
        # Compute bottom brightness
        h2, w2 = det["image"].shape[:2]
        bottom = cv2.cvtColor(det["image"][int(h2*0.68):, :], cv2.COLOR_BGR2GRAY)
        bright = (bottom > 180).sum() / bottom.size
        print(f"    is_card_back={back}  bottom_bright_ratio={bright:.3f}")

    # Save annotated images
    base, ext = os.path.splitext(image_path)
    candidates_path = base + "_candidates" + ext
    result_path     = base + "_result" + ext
    edges_path      = base + "_edges" + ext

    cv2.imwrite(candidates_path, debug_frame)
    cv2.imwrite(result_path,     out_frame)
    cv2.imwrite(edges_path,      edges)

    print(f"\nSaved:")
    print(f"  Edges:      {edges_path}")
    print(f"  Candidates: {candidates_path}  (yellow = passed area+aspect)")
    print(f"  Result:     {result_path}       (green = final detections)")

    if len(results) == 0:
        print("\nNO CARDS DETECTED. Possible causes:")
        if candidates == 0:
            print("  → No contours passed area+aspect filters.")
            print("    Check the edges image — are card edges visible?")
            print(f"    MIN_AREA={MIN_AREA}, frame_area={frame_area}")
        else:
            print(f"  → {candidates} candidate(s) found but all filtered out.")
            print("    Check the crop images to see what was detected.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: .venv/bin/python3 debug_detect.py path/to/image.jpg")
        sys.exit(1)
    main(sys.argv[1])
