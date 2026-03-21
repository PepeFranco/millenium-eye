"""
Web interface for the Yu-Gi-Oh card recognition engine.

Run:
    python3 app.py

Then open http://localhost:5000 in your browser.
"""

import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

from card_detector import detect_cards
from recogniser import load_index, recognise_card

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/recognize", methods=["POST"])
def recognize():
    data = request.get_data()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "invalid image"}), 400

    detections = detect_cards(frame)
    print(f"[recognize] frame={frame.shape}  cards_detected={len(detections)}", flush=True)

    results = []
    for i, det in enumerate(detections):
        match = recognise_card(det["image"])
        print(f"  card[{i}] low_conf={det['low_confidence']}  match={match['card_name'] if match else None}", flush=True)
        results.append({
            "corners": det["corners"].tolist(),
            "low_confidence": bool(det["low_confidence"]),
            "match": match,
        })

    return jsonify({"detections": results, "frame_shape": list(frame.shape)})


if __name__ == "__main__":
    load_index()
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
