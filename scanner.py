"""
Phase 5 — Live scanner.

Usage:
    python3 scanner.py                  # webcam
    python3 scanner.py --image path.jpg # static image
"""

import argparse
import queue
import threading
import time
from collections import deque

import cv2
import numpy as np

from card_detector import detect_cards
from recogniser import load_index, recognise_card

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEDUP_FRAMES   = 30
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_THICKNESS = 2

COLOR_MATCH    = (0,   255, 0)
COLOR_LOW_CONF = (0,   255, 255)
COLOR_NO_MATCH = (0,   0,   255)


# ---------------------------------------------------------------------------
# Background recognition worker
# ---------------------------------------------------------------------------
# The recognition thread picks up card images from _recog_queue, runs ORB
# matching (~700ms), and writes results into _recog_results.  The display
# loop never blocks on recognition — it just shows the latest cached result.

_recog_queue   = queue.Queue(maxsize=1)   # drop old frames, keep only latest
_recog_results = {}                        # slot_id → result dict | None
_recog_lock    = threading.Lock()


def _recognition_worker():
    while True:
        item = _recog_queue.get()
        if item is None:
            break
        slot_id, card_img = item
        result = recognise_card(card_img)
        with _recog_lock:
            _recog_results[slot_id] = result


def start_recognition_thread():
    t = threading.Thread(target=_recognition_worker, daemon=True)
    t.start()
    return t


def submit_for_recognition(slot_id, card_img):
    """Non-blocking submit. Drops the item if the worker is still busy."""
    try:
        _recog_queue.put_nowait((slot_id, card_img.copy()))
    except queue.Full:
        pass


def get_cached_result(slot_id):
    with _recog_lock:
        return _recog_results.get(slot_id)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_text_with_shadow(frame, text, x, y, scale, color, thickness):
    cv2.putText(frame, text, (x + 2, y + 2), FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_card(frame, detection, result):
    corners  = detection["corners"].astype(int)
    low_conf = detection["low_confidence"]
    color    = COLOR_MATCH if result else (COLOR_LOW_CONF if low_conf else COLOR_NO_MATCH)

    cv2.polylines(frame, [corners], isClosed=True, color=color, thickness=3)

    if result:
        name   = result["card_name"]
        conf   = f"{result['confidence']:.0%}"
        card_w = int(np.linalg.norm(corners[1] - corners[0]))
        scale  = max(0.8, min(2.0, card_w / 200))

        (tw, th), _ = cv2.getTextSize(name, FONT, scale, FONT_THICKNESS)
        cx = int(corners[0][0] + card_w / 2 - tw / 2)
        cy = max(int(corners[0][1]) - 12, th + 4)
        draw_text_with_shadow(frame, name, cx, cy, scale, color, FONT_THICKNESS)

        (cw, _), _ = cv2.getTextSize(conf, FONT, scale * 0.6, FONT_THICKNESS)
        draw_text_with_shadow(frame, conf, cx + tw // 2 - cw // 2, cy + th + 4,
                              scale * 0.6, color, FONT_THICKNESS)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class RecentMatches:
    def __init__(self, window=DEDUP_FRAMES):
        self.window  = window
        self.history = deque()
        self.frame   = 0

    def tick(self):
        self.frame += 1
        while self.history and self.frame - self.history[0][0] > self.window:
            self.history.popleft()

    def seen(self, card_id):
        return any(cid == card_id for _, cid in self.history)

    def mark(self, card_id):
        self.history.append((self.frame, card_id))


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def process_frame(frame, recent, frame_no):
    detections = detect_cards(frame)
    recent.tick()

    for i, det in enumerate(detections):
        slot_id = i  # simple slot: slot 0 = first card, slot 1 = second, etc.

        # Submit to background worker (non-blocking)
        submit_for_recognition(slot_id, det["image"])

        # Use last known result for this slot
        result = get_cached_result(slot_id)
        draw_card(frame, det, result)

        if result and not recent.seen(result["card_id"]):
            recent.mark(result["card_id"])
            print(
                f"  MATCH  {result['card_name']}"
                f"  |  id={result['card_id']}"
                f"  |  confidence={result['confidence']:.1%}"
                f"  |  matches={result['good_matches']}"
            )

    return frame


def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (device 0)")

    start_recognition_thread()
    print("Scanner running — press Q to quit")
    recent    = RecentMatches()
    prev_time = time.time()
    frame_no  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = process_frame(frame, recent, frame_no)
        frame_no += 1

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(annotated, f"{fps:.1f} fps", (10, 25),
                    FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("YGO Scanner", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(path):
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    start_recognition_thread()
    recent = RecentMatches()

    # For static image: submit and wait for result
    detections = detect_cards(frame)
    for i, det in enumerate(detections):
        submit_for_recognition(i, det["image"])

    time.sleep(2.0)  # give worker time to finish

    annotated = process_frame(frame, recent, 0)
    cv2.imshow("YGO Scanner", annotated)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yu-Gi-Oh card scanner")
    parser.add_argument("--image", metavar="PATH")
    args = parser.parse_args()

    load_index()

    if args.image:
        run_image(args.image)
    else:
        run_webcam()
