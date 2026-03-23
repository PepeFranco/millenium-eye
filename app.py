"""
Web interface for the Yu-Gi-Oh card recognition engine.

Development:
    .venv/bin/python3 app.py

Production:
    gunicorn -c gunicorn.conf.py app:app
"""

import os
import time
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, abort
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
@limiter.limit("600 per minute")
def recognize():
    check_token()

    t0 = time.perf_counter()

    data = request.get_data()
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


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
