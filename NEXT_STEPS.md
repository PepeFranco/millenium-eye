# Millennium Eye — Next Steps

## What works now

- Full pipeline: webcam → card detection → ORB recognition → card name on screen
- ORB + FLANN matching correctly identifies cards from real-world photos
- Background threading keeps camera loop smooth while recognition runs (~700ms per card)
- Detection handles held cards, plastic sleeves, and slight tilt
- Deduplication suppresses repeated matches within 30 frames

## Known issues to fix next

### 1. Recognition speed (highest priority)
- FLANN matching takes ~700ms per card against 7M descriptors
- **Fix**: Build a two-stage pipeline — colour histogram coarse filter to get top-200 candidates, then ORB only against those. Or switch to a bag-of-visual-words index.
- Alternative: reduce `ORB_FEATURES` from 500 → 150 in `build_database.py` and regenerate `data/orb_descriptors.npy`. Re-test accuracy to find the minimum that still reliably matches.

### 2. Card detection reliability
- Detection is inconsistent when card is not perfectly lit or background is cluttered
- The minAreaRect crop sometimes grabs too much background
- **Fix**: try adaptive thresholding in `card_detector.py` instead of fixed Canny (50/150). Also experiment with MSER region detection as an alternative to contour-based detection.

### 3. Multiple cards per frame (binder scanning)
- The recognition worker only has one queue slot — simultaneous cards overwrite each other
- **Fix**: expand `_recog_queue` and `_recog_results` to handle N slots keyed by detection bounding box position, not just index.

### 4. Startup time
- Loading the FLANN index takes ~18 seconds
- **Fix**: pre-build and pickle the FLANN index, or use a lighter index structure. Alternatively cache the trained matcher with `cv2.FlannBasedMatcher` write to disk once it supports serialisation.

### 5. Wants list / alert system
- The plan calls for alerting when a card on the user's wants list is detected
- **Fix**: add a `wants.txt` file (one card name per line), load it at startup, and trigger an audio alert (via `playsound` or `subprocess` + `afplay` on macOS) when a match is found.

### 6. Tests (Phase 6)
- `tests/` folder exists but is empty
- Need: `test_detector.py`, `test_recogniser.py`, `test_hash_db.py` as specified in `plan.md`
- Fixtures: download a handful of known card images from YGOPRODECK into `tests/fixtures/`

## File map

```
build_database.py   — downloads card images, computes ORB descriptors
card_detector.py    — finds card regions in a frame (contour + minAreaRect)
recogniser.py       — ORB + FLANN matching against the descriptor index
scanner.py          — main entry point, webcam loop, background threading
debug_detector.py   — visual tuning tool for card_detector thresholds
plan.md             — original full spec (Phases 1–6)
data/
  cards.json            — full YGOPRODECK card list
  orb_card_names.json   — card_id → card_name mapping
  orb_card_ids.npy      — descriptor-row → card_id mapping (committed)
  images/               — downloaded card images (gitignored, ~3GB)
  orb_descriptors.npy   — 7M × 32 ORB descriptors (gitignored, 216MB)
```

## To regenerate the ORB index from scratch

```bash
python3 build_database.py   # skips download if images/ exists, recomputes descriptors
```

## Why ORB instead of phash

The original plan used perceptual hashing (phash). During development, phash gave hamming distances of ~130/256 (essentially random) when comparing real camera photos to the clean digital renders in the YGOPRODECK database. ORB feature matching gave 21 good matches on the same pair. The phash approach is kept in the codebase skeleton but is no longer used.
