# Millennium Eye — Yu-Gi-Oh Card Scanner

A web app that uses your phone or laptop camera to recognise Yu-Gi-Oh cards. Point the camera at cards on a table or in a binder and the app will attempt to identify them.

Built for a local Yu-Gi-Oh community to use as a binder scanner at conventions: scan someone's binder, and the app highlights any card that appears on your wishlist.

---

## How it works

1. Open the app in your browser (phone or desktop)
2. Allow camera access when prompted
3. Point the camera at any Yu-Gi-Oh card
4. The app draws a coloured border around every card it detects and shows the name in the sidebar
5. Cards on your wishlist are highlighted in **gold** with the name of who wants them and their preferred rarity/set

> The app recognises cards by matching visual features against a database of ~12,000 card images. Results vary depending on lighting, glare, and card condition — see the known limitations section below.

---

## Screenshots

<!-- Scanner view (desktop) -->
> **TODO:** Add screenshot of the scanner with a card detected and name shown in sidebar

<!-- Scanner view (mobile, portrait) -->
> **TODO:** Add screenshot of the mobile portrait layout

<!-- Wishlist page -->
> **TODO:** Add screenshot of the wishlist management page

---

## Getting started (as a user)

### What you need

- A phone, tablet, or laptop with a camera and a modern browser (Chrome, Firefox, Safari)
- The **access token** for your community's scanner — ask whoever runs it

### Step 1 — Open the app

Go to the URL your community uses (e.g. `https://millenium-eye.yourdomain.com`).

You will see a password prompt on first visit.

### Step 2 — Enter the access token

The app is private — only people with the token can use it. Enter the token you received and press **Enter**. The token is saved in your browser so you only need to do this once per device.

### Step 3 — Allow camera access

Your browser will ask permission to use the camera. Tap **Allow**. If you accidentally deny it, go to your browser settings, find the site, and enable camera access.

### Step 4 — Scan cards

Point your camera at any Yu-Gi-Oh cards. Hold it steady for a moment — the app scans several frames per second.

- **Green border** — card recognised
- **Yellow border** — card detected but confidence is low (try moving closer or improving the light)
- **Red border** — card shape detected but not matched

The card name appears in the panel on the right (desktop) or the bottom half of the screen (phone).

### Step 5 — Set up your wishlist

Tap the **Wishlist** link in the sidebar (or go to `/wishlist` directly) to manage the cards you're looking for.

For each entry you can save:
- Your name
- The card name (with autocomplete — start typing and pick from the list)
- Preferred rarity (e.g. *Ultra Rare*, *Secret Rare*) — optional
- Preferred set (e.g. *LOB*, *KEKM*) — optional

When the scanner spots a card that is on anyone's wishlist, its border turns **gold** and the sidebar shows who wants it and what version they're after.

---

## Tips for best results

- **Lighting matters** — natural light or a bright overhead light works best. Avoid direct glare on shiny cards or sleeves.
- **Keep it flat** — cards lying flat on a table are easier to detect than cards held at an angle.
- **Distance** — hold the camera about 20–40 cm away. Too close and the whole card won't fit; too far and detail is lost.
- **Multiple cards** — the app can detect and identify several cards at once.

---

## For the person running the server

### Requirements

- Python 3.10+
- See `requirements.txt` for Python dependencies
- OpenCV (`opencv-python`)

### First-time setup

```bash
# Clone the repo
git clone https://github.com/PepeFranco/millenium-eye.git
cd millenium-eye

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build the card database (downloads ~12,000 images, takes ~30 min)
python3 build_database.py

# Run locally
python3 app.py
# Open http://localhost:5001
```

### Environment variables

| Variable | Description |
|---|---|
| `ACCESS_TOKEN` | Shared secret token required to use the app. If not set, auth is skipped (dev mode). |

Generate a secure token:
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Production deployment

See [`DEPLOY.md`](DEPLOY.md) for the full guide to deploying on a Hetzner CX22 server with Gunicorn, Nginx, and HTTPS via Let's Encrypt.

### Deploying updates

If you set up the server alias from DEPLOY.md, just run:
```bash
deploy-eye
```

Otherwise:
```bash
cd /opt/millenium-eye && git pull && systemctl restart millenium-eye
```

---

## Project structure

```
millenium-eye/
├── app.py                  # Flask web server, API routes
├── card_detector.py        # Detects card regions in a camera frame
├── recogniser.py           # Matches card images against the ORB index
├── wishlist.py             # SQLite wishlist store
├── build_database.py       # Downloads card images and builds the ORB index
├── scanner.py              # Original desktop scanner (OpenCV window)
├── templates/
│   ├── index.html          # Scanner UI
│   └── wishlist.html       # Wishlist management page
├── gunicorn.conf.py        # Production server config
├── millenium-eye.service   # systemd unit file
├── nginx.conf              # Nginx reverse proxy config
├── DEPLOY.md               # Step-by-step deployment guide
└── data/                   # Generated — not committed
    ├── orb_descriptors.npy
    ├── orb_card_ids.npy
    ├── orb_card_names.json
    ├── wishlist.db
    └── images/             # Downloaded card images
```

---

## Known limitations

- **Glare on sleeves** — reflective card sleeves can confuse the detector. Cross-polarised lighting is the proper fix; for now, tilt the card slightly to reduce glare.
- **Similar-looking cards** — cards with nearly identical artwork (e.g. alternate arts or reprints) may occasionally be confused with each other.
- **Speed** — recognition takes ~300–700 ms per frame depending on server load. The UI stays responsive while the server processes.
- **Card backs** — the detector filters out card backs automatically.

---

## Acknowledgements

Card data and images provided by [YGOPRODECK](https://ygoprodeck.com/).
