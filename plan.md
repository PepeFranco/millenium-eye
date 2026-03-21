## Card Recognition Engine — Claude Code Instructions

---

### Project goal

Build a Yu-Gi-Oh card recognition engine that takes a video feed from a webcam or phone camera, detects Yu-Gi-Oh cards in frame, identifies each card by matching it against the YGOPRODECK card image database, and returns the card name and ID in real time. Deployed as a web app for a local Yu-Gi-Oh community (~20 users) to use as a convention binder scanner that alerts users when a card on their wants list is detected.

---

### What has been built

- **`build_database.py`** — fetches all card images from YGOPRODECK, computes ORB descriptors for each card, stores them in `data/orb_descriptors.npy`, `data/orb_card_ids.npy`, `data/orb_card_names.json`
- **`card_detector.py`** — detects card regions in a BGR frame using contour detection, deskews them to 200×290px, filters card backs and glare
- **`recogniser.py`** — loads ORB descriptors into a FLANN-LSH index at startup, matches query card images by voting on nearest-neighbour descriptors
- **`scanner.py`** — original desktop entry point (OpenCV window), kept for local testing
- **`app.py`** — Flask web server with `POST /api/recognize` endpoint, token auth (`X-Access-Token` header), rate limiting (60 req/min per IP via Flask-Limiter)
- **`templates/index.html`** — browser UI: getUserMedia camera feed, canvas bounding box overlay, matched cards sidebar, token gate on first visit
- **`gunicorn.conf.py`** — production server config: 2 workers × 8 gthreads, `preload_app=True` so ORB index loads once
- **`millenium-eye.service`** — systemd unit file
- **`nginx.conf`** — reverse proxy config, HTTPS-ready via Certbot
- **`DEPLOY.md`** — full Hetzner CX22 deployment guide

---

### Technical decisions made

- **ORB + FLANN** (not phash) — switched from perceptual hashing to ORB feature matching for better accuracy on similar-looking cards
- **Background recognition thread** in scanner.py — recognition runs async so the display loop never blocks
- **Token auth** — single shared `ACCESS_TOKEN` env var, stored in browser localStorage, no user accounts
- **Hetzner CX22** (~€3.79/mo) chosen over AWS (too expensive) and Vercel (serverless, wrong architecture)
- **Gunicorn preload** — ORB index (~200MB) loaded once in master process, shared across workers via fork

---

### Current state

The web app works end-to-end locally. Deployed to Hetzner CX22 — database build in progress on the server.

---

### Next: complete server deployment

Follow `DEPLOY.md`. Remaining steps after database build finishes:

1. Fix permissions: `chown -R www-data:www-data /opt/millenium-eye`
2. Generate access token: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`
3. Set token in service file: `nano /opt/millenium-eye/millenium-eye.service` → replace `CHANGE_ME`
4. Install and start service:
   ```bash
   cp millenium-eye.service /etc/systemd/system/
   systemctl daemon-reload
   systemctl enable millenium-eye
   systemctl start millenium-eye
   ```
5. Configure Nginx: replace `YOUR_DOMAIN_OR_IP` in `nginx.conf`, copy to sites-available, enable
6. Run Certbot for HTTPS: `certbot --nginx -d YOUR_DOMAIN`
7. Set up firewall: allow SSH (your IP only), 80, 443

---

### Next feature: wants list

The original goal is to alert users when a card on their wants list appears in frame. This is not yet built.

**Proposed approach:**

- Add a **wants list** UI to the sidebar — a text area or search box where users paste card names (one per line)
- Store the list in `localStorage` so it persists between sessions
- On each match, check if the matched `card_name` is in the wants list
- If it is, highlight the match differently in the UI (e.g. gold border, notification sound, push notification)
- The wants list is per-user (client-side only) — no server changes needed

---

### Known issues

- **Glare on binder sleeves** — flagged as `low_confidence` but not fully solved. Future improvement: cross-polarised lighting.
- **Multiple printings** — recogniser returns the specific printing detected (correct behaviour, card_id distinguishes them).
- **Card backs** — filtered by dominant colour heuristic in `card_detector.py`.
- **Speed** — ORB matching takes ~700ms. Recognition runs in a background thread / async so UI stays responsive.
