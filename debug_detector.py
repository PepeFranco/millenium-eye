"""
Debug tool for tuning card detection.

Shows a live window with:
  - Original frame with detected cards outlined
  - Edge detection output
  - Terminal logs explaining why each contour was rejected

Usage:
    python3 debug_detector.py              # webcam
    python3 debug_detector.py --image x.jpg
"""

import argparse
import cv2
import numpy as np

MIN_AREA       = 8000
MAX_AREA_RATIO = 0.80
ASPECT_MIN     = 0.3
ASPECT_MAX     = 2.5


def analyse_frame(frame):
    h, w = frame.shape[:2]
    frame_area = h * w

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    overlay = frame.copy()
    rejected = []
    accepted = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue  # too small to be worth logging
        if area > MAX_AREA_RATIO * frame_area:
            rejected.append(f"area={area:.0f} > max ({MAX_AREA_RATIO*frame_area:.0f}) — too large")
            continue

        hull   = cv2.convexHull(cnt)
        peri   = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.06 * peri, True)
        if len(approx) != 4:
            rejected.append(f"area={area:.0f}  vertices={len(approx)} — not a quad")
            cv2.drawContours(overlay, [approx], -1, (0, 0, 180), 1)
            continue

        x, y, cw, ch = cv2.boundingRect(approx)
        aspect = cw / ch if ch else 0
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            rejected.append(
                f"area={area:.0f}  aspect={aspect:.3f} "
                f"(need {ASPECT_MIN}–{ASPECT_MAX}) — wrong shape"
            )
            cv2.drawContours(overlay, [approx], -1, (0, 140, 255), 1)
            continue

        # accepted
        cv2.drawContours(overlay, [approx], -1, (0, 255, 0), 2)
        cv2.putText(overlay, f"a={area:.0f} ar={aspect:.2f}",
                    (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        accepted += 1

    return overlay, edges_dilated, accepted, rejected


def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    print("Debug mode — press Q to quit")
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        overlay, edges, accepted, rejected = analyse_frame(frame)

        # stack original + edges side by side
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        h = min(overlay.shape[0], edges_bgr.shape[0])
        combined = np.hstack([
            cv2.resize(overlay,    (int(overlay.shape[1]    * h / overlay.shape[0]),    h)),
            cv2.resize(edges_bgr,  (int(edges_bgr.shape[1]  * h / edges_bgr.shape[0]), h)),
        ])
        cv2.putText(combined, f"accepted: {accepted}  rejected (shown): {len(rejected)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Debug — left: overlay | right: edges", combined)

        # Print rejection log every 30 frames so you can read it in the terminal
        if frame_no % 30 == 0:
            print(f"\n--- frame {frame_no}  accepted={accepted}  rejected={len(rejected)} ---")
            for r in rejected:
                print(" ", r)
        frame_no += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(path):
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(path)

    overlay, edges, accepted, rejected = analyse_frame(frame)

    print(f"\nAccepted: {accepted}")
    print(f"Rejected: {len(rejected)}")
    for r in rejected:
        print(" ", r)

    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    h = min(overlay.shape[0], edges_bgr.shape[0])
    combined = np.hstack([
        cv2.resize(overlay,   (int(overlay.shape[1]   * h / overlay.shape[0]),   h)),
        cv2.resize(edges_bgr, (int(edges_bgr.shape[1] * h / edges_bgr.shape[0]), h)),
    ])
    cv2.imshow("Debug — left: overlay | right: edges", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", metavar="PATH")
    args = parser.parse_args()

    if args.image:
        run_image(args.image)
    else:
        run_webcam()
