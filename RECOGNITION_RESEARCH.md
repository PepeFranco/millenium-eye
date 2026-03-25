# Card Recognition — Approaches Tried

## The Problem

Match a camera photograph of a physical Yu-Gi-Oh card against a database of
~3,800 reference scans downloaded from YGOPRODECK. Constraints:

- Reference images are small thumbnails (~100×145 px)
- Query images are camera frames: JPEG-compressed, variable lighting, possible
  glare from holographic foil, card sleeves, slight perspective distortion
- Must run in real time on a modest server (no GPU)

---

## Approach 1 — ORB + FLANN-LSH (200 features, 14,000 cards)

**What it is:** OpenCV's ORB keypoint detector extracts 200 binary descriptors
per image. All descriptors are loaded into a FLANN-LSH index. For each query
frame, descriptors are matched against the full index and votes are counted per
card ID.

**Why it failed:**
- 14,000 cards × 200 features = 2.8M descriptors in the index. With so many
  cards, the nearest-neighbour matches are almost always wrong cards.
- The Lowe ratio test rejects most matches, leaving too few votes for any card
  to cross the threshold.
- Slow: ~80–300ms per card recognised.

---

## Approach 2 — ORB + FLANN-LSH (200 features, 3,817 cards, CLAHE)

**Changes:** Added CLAHE contrast enhancement, cut database to cards released
before 2010-04-26 (reducing from 14,000 to 3,817 cards), reduced ORB features
to 200.

**Why it failed:**
- Detection improved but recognition was still unreliable.
- With only 200 features per card, there weren't enough matching points to vote
  confidently, especially for cards with large uniform areas (sky, simple
  backgrounds) that produce few keypoints.
- CLAHE helped slightly but not enough to overcome the low feature count.

---

## Approach 3 — MobileNetV3-Small CNN Embeddings

**What it is:** Pretrained MobileNetV3-Small (ImageNet weights) used as a
feature extractor. Each card image is passed through the network to produce a
576-dimensional embedding. Recognition = cosine similarity search against all
stored embeddings.

**Why it failed:**
- MobileNetV3 was trained on ImageNet (dogs, cats, furniture, etc.). It has no
  concept of Yu-Gi-Oh card artwork. All card images look roughly the same to
  it — colourful illustrated rectangles — so embeddings cluster tightly and
  are not discriminative.
- Cosine similarity scores of 0.60–0.85 for both correct and incorrect cards,
  with no reliable threshold separating them.
- Produced many false positives at low thresholds; no detections at high ones.
- Added a heavy dependency (PyTorch ~200MB) for no benefit.

---

## Approach 4 — Perceptual Hash (pHash) on Artwork Region

**What it is:** Crop the artwork area of the card (top 13%–57%, inset from the
border) and compute a 64-bit pHash (DCT-based). Recognition = minimum Hamming
distance against all stored hashes.

**Why it failed:**
- Reference images are ~100×145 px thumbnails. After cropping the artwork
  region, we have roughly 58×86 px. pHash resizes this to 8×8 for DCT — far
  too little information to discriminate between 3,817 cards.
- A camera photograph of a physical card produces very different DCT frequency
  content than a clean digital scan, even for the same card. Hamming distances
  of 16–20 for everything — correct cards and random background regions alike.
- No useful separation between true matches and non-matches.

---

## Approach 5 — ORB + FLANN-LSH (500 features, 3,817 cards) — Current

**Changes:** Back to ORB but with 500 features per card and the filtered
3,817-card database.

**Rationale:** ORB is the correct class of algorithm for photo-vs-reference
matching (it is scale- and rotation-invariant, binary descriptors, fast). The
earlier ORB failure was a scale problem (14,000 cards), which is now fixed.
500 features gives 3× more matching points than the 200-feature attempt,
improving the vote signal especially for cards with complex artwork.

**Status:** In testing.

---

## Root Causes and Known Hard Problems

1. **Reference image quality.** All approaches are limited by the 100×145 px
   YGOPRODECK small thumbnails. Downloading full-size images (420×610 px)
   would give any method more to work with. A future improvement would be to
   use `image_url` instead of `image_url_small` in `build_database.py`.

2. **Photo vs. scan mismatch.** Camera images have lighting variation, glare
   (especially on holographic rares), motion blur, and sleeve reflections.
   Reference images are clean digital scans. This domain gap is the hardest
   problem to solve without training data.

3. **Card detection.** The Canny edge detector finds card-like rectangles
   inconsistently — missing real cards sometimes, firing on background objects
   other times. Better detection (e.g. a trained YOLO model) would help every
   recognition approach downstream.

4. **The ceiling.** Without a model fine-tuned on photographs of physical
   Yu-Gi-Oh cards, recognition accuracy will remain limited. The practical
   path to a reliable solution is either: (a) a larger training dataset of
   card photos + fine-tuned CNN, or (b) OCR on the card name, which is high-
   contrast printed text and robust to the problems above.
