"""
Web interface for the Yu-Gi-Oh card recognition engine.

Development:
    .venv/bin/python3 app.py

Production:
    gunicorn -c gunicorn.conf.py app:app
"""

import datetime
import os
import shutil
import time
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, abort, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from card_detector import detect_cards
from recogniser import load_index, recognise_card, get_valid_card_names
from wishlist import init_db, get_all as wl_get_all, add_entry as wl_add, remove_entry as wl_remove

app = Flask(__name__)

# Load the ORB index at import time so Gunicorn's preload_app shares it
# across workers. Also works when running with `python3 app.py` directly.
load_index()
init_db()

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per minute"],
    storage_uri="memory://",
)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "")


def check_token():
    if not ACCESS_TOKEN:
        return  # token not configured — skip auth (dev mode)
    token = request.headers.get("X-Access-Token", "")
    if token != ACCESS_TOKEN:
        abort(401)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/wishlist")
def wishlist_page():
    return render_template("wishlist.html")


@app.route("/api/recognize", methods=["POST"])
@limiter.exempt
def recognize():
    check_token()

    t0 = time.perf_counter()

    data = request.get_data()
    if not data:
        return jsonify({"error": "empty body"}), 400
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "invalid image"}), 400
    t_decode = time.perf_counter()

    detections = detect_cards(frame)
    t_detect = time.perf_counter()

    results = []
    for det in detections:
        match = recognise_card(det["image"])
        results.append({
            "corners": det["corners"].tolist(),
            "low_confidence": bool(det["low_confidence"]),
            "match": match,
        })
    t_recognise = time.perf_counter()

    print(
        f"[timing] decode={1000*(t_decode-t0):.0f}ms  "
        f"detect={1000*(t_detect-t_decode):.0f}ms  "
        f"recognise={1000*(t_recognise-t_detect):.0f}ms  "
        f"total={1000*(t_recognise-t0):.0f}ms  "
        f"cards={len(detections)}",
        flush=True,
    )

    return jsonify({"detections": results})


@app.route("/api/cards")
def card_names():
    check_token()
    return jsonify(get_valid_card_names())


@app.route("/api/wishlist", methods=["GET"])
def wishlist_get():
    check_token()
    return jsonify(wl_get_all())


@app.route("/api/wishlist", methods=["POST"])
def wishlist_post():
    check_token()
    body   = request.get_json(silent=True) or {}
    player = (body.get("player_name") or "").strip()
    card   = (body.get("card_name") or "").strip()
    if not player or not card:
        return jsonify({"error": "player_name and card_name required"}), 400
    valid = set(get_valid_card_names())
    if valid and card not in valid:
        return jsonify({"error": "unknown card name"}), 400
    entry_id = wl_add(
        player, card,
        (body.get("preferred_rarity") or "").strip() or None,
        (body.get("preferred_set") or "").strip() or None,
    )
    return jsonify({"id": entry_id}), 201


@app.route("/api/wishlist/<int:entry_id>", methods=["DELETE"])
def wishlist_delete(entry_id):
    check_token()
    wl_remove(entry_id)
    return "", 204


TRAINING_DIR = os.path.join(os.path.dirname(__file__), "data", "training_samples")
_TRAINING_REAL = None  # resolved lazily after first mkdir


def _training_real():
    global _TRAINING_REAL
    if _TRAINING_REAL is None:
        os.makedirs(TRAINING_DIR, exist_ok=True)
        _TRAINING_REAL = os.path.realpath(TRAINING_DIR)
    return _TRAINING_REAL


def _safe_path(rel):
    """Return absolute path only if it stays inside TRAINING_DIR."""
    full = os.path.realpath(os.path.join(TRAINING_DIR, rel))
    if not full.startswith(_training_real() + os.sep):
        abort(400)
    return full


@app.route("/training-images")
def training_images_page():
    check_token()
    return render_template("training_images.html")


@app.route("/api/training-samples")
def training_samples_list():
    check_token()
    if not os.path.isdir(TRAINING_DIR):
        return jsonify([])
    results = []
    for dirpath, _, filenames in os.walk(TRAINING_DIR):
        for filename in sorted(filenames):
            if not filename.endswith(".jpg"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, filename), TRAINING_DIR).replace("\\", "/")
            incorrect = rel.startswith("incorrect/")
            card_id   = rel.split("/")[0] if not incorrect else rel.split("/")[1].split("_", 1)[0]
            results.append({"path": rel, "card_id": card_id, "incorrect": incorrect})
    results.sort(key=lambda x: x["path"])
    return jsonify(results)


@app.route("/api/training-samples/<path:filename>", methods=["DELETE"])
def training_sample_delete(filename):
    check_token()
    filepath = _safe_path(filename)
    if not os.path.isfile(filepath):
        abort(404)
    os.remove(filepath)
    # Clean up empty parent directory (but not TRAINING_DIR itself)
    parent = os.path.dirname(filepath)
    if parent != os.path.realpath(TRAINING_DIR) and os.path.isdir(parent) and not os.listdir(parent):
        os.rmdir(parent)
    return "", 204


@app.route("/api/training-samples", methods=["DELETE"])
def training_samples_delete_all():
    check_token()
    if os.path.isdir(TRAINING_DIR):
        shutil.rmtree(TRAINING_DIR)
    os.makedirs(TRAINING_DIR, exist_ok=True)
    print("[training] deleted all training samples", flush=True)
    return "", 204


@app.route("/training-samples/<path:filename>")
def training_sample_image(filename):
    check_token()
    _safe_path(filename)  # security check
    return send_from_directory(TRAINING_DIR, filename)


@app.route("/api/training-sample", methods=["POST"])
def training_sample_save():
    check_token()
    card_id = (request.args.get("card_id") or "").strip()
    if not card_id:
        return jsonify({"error": "card_id required"}), 400

    data = request.get_data()
    if not data:
        return jsonify({"error": "empty body"}), 400

    out_dir = os.path.join(TRAINING_DIR, card_id)
    os.makedirs(out_dir, exist_ok=True)

    ts       = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{ts}.jpg"
    with open(os.path.join(out_dir, filename), "wb") as f:
        f.write(data)

    saved_path = f"{card_id}/{filename}"
    print(f"[training] saved: {saved_path}", flush=True)
    return jsonify({"path": saved_path}), 201


@app.route("/api/training-sample/mark-incorrect", methods=["POST"])
def training_sample_mark_incorrect():
    check_token()
    body = request.get_json(silent=True) or {}
    path = (body.get("path") or "").strip()
    if not path or path.startswith("incorrect/"):
        return jsonify({"error": "invalid path"}), 400

    src = _safe_path(path)
    if not os.path.isfile(src):
        abort(404)

    # Move to incorrect/{card_id}_{original_filename}
    parts    = path.split("/")
    card_id  = parts[0]
    basename = parts[-1]

    incorrect_dir = os.path.join(TRAINING_DIR, "incorrect")
    os.makedirs(incorrect_dir, exist_ok=True)
    os.rename(src, os.path.join(incorrect_dir, f"{card_id}_{basename}"))

    # Remove empty card directory
    card_dir = os.path.join(TRAINING_DIR, card_id)
    if os.path.isdir(card_dir) and not os.listdir(card_dir):
        os.rmdir(card_dir)

    print(f"[training] marked incorrect: {path}", flush=True)
    return "", 204


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
