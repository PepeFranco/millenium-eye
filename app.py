"""
Web interface for the Yu-Gi-Oh card recognition engine.

Development:
    .venv/bin/python3 app.py

Production:
    gunicorn -c gunicorn.conf.py app:app
"""

import os
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from card_detector import detect_cards
from recogniser import load_index, recognise_card

app = Flask(__name__)

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


@app.route("/api/recognize", methods=["POST"])
@limiter.limit("60 per minute")
def recognize():
    check_token()

    data = request.get_data()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "invalid image"}), 400

    detections = detect_cards(frame)

    results = []
    for det in detections:
        match = recognise_card(det["image"])
        results.append({
            "corners": det["corners"].tolist(),
            "low_confidence": bool(det["low_confidence"]),
            "match": match,
        })

    return jsonify({"detections": results})


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_index()
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
