Here's a detailed instruction document you can paste directly into Claude Code:

---

## Card Recognition Engine — Claude Code Instructions

---

### Project goal

Build a Yu-Gi-Oh card recognition engine in Python that takes a video feed from a webcam or phone camera, detects Yu-Gi-Oh cards in frame, identifies each card by matching it against the YGOPRODECK card image database, and returns the card name and ID in real time. This will be the foundation for a convention binder scanner that alerts users when a card on their wants list is detected.

---

### Technical approach

Use perceptual hashing for card matching. Do not use a trained ML classifier. The approach is:

1. Pull the full card image database from the YGOPRODECK API
2. Pre-compute a perceptual hash for every card image and store it locally
3. At scan time, capture frames from the camera
4. Detect card regions in each frame using OpenCV contour detection
5. Crop, deskew and normalise each detected card region
6. Compute the perceptual hash of the detected card
7. Find the nearest match in the hash database using Hamming distance
8. Return the card name, ID, and confidence score if above threshold

---

### Phase 1 — Environment setup

Set up a Python virtual environment. Install the following dependencies:

- `opencv-python` — camera capture and image processing
- `imagehash` — perceptual hashing (use `phash` specifically, not `ahash`)
- `Pillow` — image handling required by imagehash
- `requests` — for YGOPRODECK API calls
- `numpy` — array operations
- `tqdm` — progress bars for the database build step
- `sqlite3` — ships with Python stdlib, used for the hash database

Create a `requirements.txt` with pinned versions.

---

### Phase 2 — Build the card hash database

Create a script called `build_database.py`.

**Step 1 — Fetch card list**

Call the YGOPRODECK API endpoint:
```
https://db.ygoprodeck.com/api/v7/cardinfo.php
```
This returns a JSON array of every card in the game. Each card object contains:
- `id` — numeric card ID
- `name` — card name string
- `card_images` — array of image objects, each with `image_url` and `image_url_small`

Parse the response and extract id, name, and image_url for every card. There are approximately 12,000–15,000 cards. Store the raw card list as `data/cards.json`.

**Step 2 — Download card images**

Download the image for every card from `image_url` into a local folder `data/images/{card_id}.jpg`. Use `image_url_small` not the full size — the small images are around 300×440px which is sufficient for hashing and much faster to download. 

Implement resumable downloading — check if the file already exists before downloading. Use a short sleep (0.05s) between requests to be respectful of the API. Wrap in a try/except so a single failed download does not abort the whole process. Log any failed downloads to `data/failed_downloads.txt` for retry.

**Step 3 — Compute perceptual hashes**

For every downloaded image, compute a perceptual hash using `imagehash.phash()` with `hash_size=16` (giving a 256-bit hash for better accuracy than the default 64-bit). Store results in a SQLite database at `data/hashes.db` with the following schema:

```sql
CREATE TABLE card_hashes (
    card_id INTEGER PRIMARY KEY,
    card_name TEXT NOT NULL,
    phash TEXT NOT NULL
);
```

Store the hash as its string representation. Also build an in-memory structure for fast lookup — a list of `(card_id, card_name, hash_object)` tuples that can be loaded at runtime.

**Step 4 — Export a fast lookup file**

After building the database, export a pickle file `data/hash_index.pkl` containing the list of `(card_id, card_name, phash_object)` tuples. This is what the recognition engine loads at startup — loading from pickle is much faster than rebuilding from SQLite on every run.

---

### Phase 3 — Card detection in frame

Create a module `card_detector.py`.

The goal is to find rectangular card-shaped regions in a camera frame and return them as cropped, deskewed images ready for hashing.

**Detection pipeline:**

1. Convert frame to greyscale
2. Apply Gaussian blur (kernel 5×5) to reduce noise
3. Apply Canny edge detection with thresholds 50 and 150
4. Dilate edges (3×3 kernel, 2 iterations) to close gaps
5. Find contours using `cv2.findContours` with `RETR_EXTERNAL` and `CHAIN_APPROX_SIMPLE`
6. For each contour, compute the approximate polygon using `cv2.approxPolyDP` with epsilon = 0.02 × arc length
7. Filter for quadrilaterals only (exactly 4 vertices)
8. Filter by area — discard any contour whose bounding area is less than 8000 pixels or greater than 80% of total frame area
9. Filter by aspect ratio — Yu-Gi-Oh cards are 59mm × 86mm (ratio ≈ 0.686). Accept contours whose bounding rect aspect ratio is between 0.55 and 0.80 to allow for perspective distortion
10. For each passing contour, apply a perspective transform to produce a deskewed rectangle of exactly 200×290 pixels (preserving card proportions)

The perspective transform should use `cv2.getPerspectiveTransform` with the four detected corner points sorted in order: top-left, top-right, bottom-right, bottom-left. Implement a `sort_corners` helper function that sorts by sum (top-left has smallest sum) and difference (top-right has smallest difference).

Return a list of cropped card images as numpy arrays.

**Important notes:**
- A binder page will have up to 9 cards visible simultaneously. The detector must handle multiple cards per frame.
- Binder sleeves cause glare. Add a glare detection step: compute the mean pixel value of the detected region. If it exceeds 240 (near-white), mark the detection as low confidence rather than discarding it — the card may still be partially readable.
- Cards in a binder are usually fairly flat but may have slight curve. The perspective transform handles mild distortion well.

---

### Phase 4 — Card recognition engine

Create a module `recogniser.py`.

**At startup:**
- Load `data/hash_index.pkl` into memory as the hash index
- Log how many cards are loaded

**Recognition function signature:**
```python
def recognise_card(card_image_bgr: np.ndarray) -> dict | None:
```

**Steps:**
1. Convert the BGR image to RGB (OpenCV uses BGR, Pillow uses RGB)
2. Convert to Pillow Image
3. Compute `imagehash.phash(image, hash_size=16)`
4. Iterate through the hash index computing Hamming distance between the query hash and every stored hash
5. Find the minimum distance match
6. If minimum distance is ≤ 15 (out of 256 bits), return a result dict:
```python
{
    "card_id": int,
    "card_name": str,
    "confidence": float,  # 1.0 - (distance / 256)
    "hamming_distance": int
}
```
7. If minimum distance is > 15, return `None` (no confident match)

**Performance note:** Iterating 12,000 hashes naively is fine for a first pass — it takes under 100ms in Python. If performance becomes an issue later, the hash index can be converted to a numpy array of uint8 packed bits and vectorised with numpy bitwise operations for a 100× speedup. Implement the naive version first, add a comment noting the optimisation path.

---

### Phase 5 — Live scanner

Create `scanner.py` as the main entry point.

This script ties everything together:

1. Load the recogniser (which loads the hash index)
2. Open the default webcam using `cv2.VideoCapture(0)`
3. Enter a capture loop:
   - Read a frame
   - Run card detection on the frame
   - For each detected card region, run recognition
   - Draw bounding boxes on the original frame — green for confident match, yellow for low confidence, red for no match
   - Overlay the card name on the bounding box if matched
   - Display the annotated frame with `cv2.imshow`
   - Print any new matches to the terminal with card name and confidence
   - Deduplicate — if the same card has been reported in the last 30 frames, don't report it again
   - Press Q to quit

Add a `--image` flag that accepts a file path instead of webcam input, for testing against static images.

---

### Phase 6 — Testing

Create a `tests/` folder.

Write tests for:
- `test_detector.py` — load a test image of a Yu-Gi-Oh card (download one from YGOPRODECK), assert that the detector finds exactly one contour, assert that the output image is 200×290px
- `test_recogniser.py` — take a known card image, run it through the recogniser, assert the correct card name is returned
- `test_hash_db.py` — assert the database contains at least 10,000 entries, assert no null hashes

Download a handful of test card images directly from YGOPRODECK and store them in `tests/fixtures/`.

---

### Project structure

```
yugioh-scanner/
├── data/
│   ├── images/          # downloaded card images (gitignored)
│   ├── cards.json       # card list from API
│   ├── hashes.db        # SQLite hash database
│   └── hash_index.pkl   # fast-load pickle index
├── tests/
│   ├── fixtures/        # test card images
│   ├── test_detector.py
│   ├── test_recogniser.py
│   └── test_hash_db.py
├── build_database.py
├── card_detector.py
├── recogniser.py
├── scanner.py
├── requirements.txt
└── README.md
```

---

### Known issues to anticipate

- **Glare on binder sleeves** is the main recognition failure mode. The detector should flag high-brightness regions. A future improvement is to add cross-polarised lighting (two polarising filters at 90°) but that is out of scope for now.
- **Multiple printings of the same card** — many cards have been reprinted with different artwork. The recogniser will return the specific printing detected, which is correct behaviour. Make sure the result dict includes the card_id so the calling code can distinguish between printings.
- **Card backs** — the detector will occasionally pick up card backs. Add a simple check: if the detected region's dominant colour is dark purple/black (the back of a YGO card), discard it before running the recogniser.
- **Speed** — the full pipeline (detect + recognise) should run comfortably at 10fps on a modern laptop. If it drops below this, add frame skipping — only run detection every other frame and interpolate bounding boxes.

---

### What success looks like

The end state of this phase is: point a webcam at a Yu-Gi-Oh card lying on a table and within one second the terminal prints the correct card name with a confidence score above 0.90. Once that works reliably on single cards, test with a binder page of 9 cards simultaneously.
