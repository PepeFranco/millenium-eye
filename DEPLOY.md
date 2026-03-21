# Deploying to Hetzner

## 1. Create the server

- Go to hetzner.com → Cloud → New Server
- Location: pick closest to you
- Image: **Ubuntu 24.04**
- Type: **CX22** (2 vCPU, 4 GB RAM, ~€3.79/mo)
- Add your SSH key

## 2. Point a domain at it (optional but recommended for HTTPS)

Add an A record pointing your domain to the server's IP.

## 3. SSH in and set up the server

```bash
ssh root@YOUR_SERVER_IP

# System deps
apt update && apt install -y python3.11 python3.11-venv python3-pip nginx certbot python3-certbot-nginx git libgl1

# App user
useradd -m -s /bin/bash appuser

# Clone repo
git clone https://github.com/PepeFranco/millenium-eye.git /opt/millenium-eye
cd /opt/millenium-eye

# Virtualenv + deps
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Build the card database (if not already done — takes ~30 min)
.venv/bin/python3 build_database.py

# Fix permissions
chown -R www-data:www-data /opt/millenium-eye
```

## 4. Generate an access token

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Copy the output — this is your `ACCESS_TOKEN`. Share it privately with your 20 members.

## 5. Configure the systemd service

```bash
# Set your token in the service file
nano /opt/millenium-eye/millenium-eye.service
# Replace CHANGE_ME with your token

cp /opt/millenium-eye/millenium-eye.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable millenium-eye
systemctl start millenium-eye

# Check it started (ORB index takes ~30s to load)
systemctl status millenium-eye
journalctl -u millenium-eye -f
```

## 6. Configure Nginx

```bash
# Replace YOUR_DOMAIN_OR_IP in the nginx config
nano /opt/millenium-eye/nginx.conf

cp /opt/millenium-eye/nginx.conf /etc/nginx/sites-available/millenium-eye
ln -s /etc/nginx/sites-available/millenium-eye /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

## 7. HTTPS with Let's Encrypt (requires a domain)

```bash
certbot --nginx -d YOUR_DOMAIN
```

Certbot will auto-update the nginx config and set up renewal.

## 8. Firewall

```bash
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw enable
```

## Updating the app

```bash
cd /opt/millenium-eye
git pull
.venv/bin/pip install -r requirements.txt
systemctl restart millenium-eye
```

## Rotating the access token

Edit the `ACCESS_TOKEN` in `/etc/systemd/system/millenium-eye.service`, then:

```bash
systemctl daemon-reload && systemctl restart millenium-eye
```

All existing sessions will be logged out and users will need to enter the new token.
