# Training Steps

## On your Mac

Run these in order from the project root.

### 1. Generate synthetic training data (~1-2 hours)

```bash
.venv/bin/python3 generate_training_data.py
```

Produces ~221,700 augmented images in data/synthetic/ (3,689 Edison cards × 60 variants).

### 2. Fine-tune the model on M1 (~1-2 hours)

```bash
.venv/bin/python3 train_model.py
```

Saves weights to data/finetuned_mobilenet.pth.
Watch the accuracy climb each epoch — expect 0.7+ by the end.

### 3. Export to ONNX

```bash
.venv/bin/python3 export_onnx.py
```

Produces data/card_embeddings.onnx (~10 MB).

### 4. Build the embedding index

```bash
.venv/bin/python3 build_database.py
```

Produces data/cnn_embeddings.npy, data/cnn_card_ids.npy, data/cnn_card_names.json.

### 5. Commit and push

```bash
git add data/card_embeddings.onnx data/class_to_card_id.json data/cnn_card_names.json data/cnn_card_ids.npy data/cnn_embeddings.npy
git commit -m "Add fine-tuned card embedding model and index"
git push
```

---

## On the server

### 6. Pull and rebuild the ORB index with correct Edison cards

```bash
ssh millenium-eye
cd /opt/millenium-eye
git pull
rm -f data/orb_descriptors.npy data/orb_card_ids.npy data/orb_card_names.json
.venv/bin/python3 build_database.py
```

This fetches edison_cards.json from YGOPRODECK and rebuilds the index
with only the 3,689 correct Edison cards. Image download is skipped
since images_full/ is already populated.

### 7. Remove non-Edison images from images_full

Now that edison_cards.json exists on the server, run the cleanup:

```bash
.venv/bin/python3 -c "
import json, os
with open('data/edison_cards.json') as f:
    cards = json.load(f)
allowed = {str(img['id']) for c in cards for img in c.get('card_images',[])}
removed = 0
for f in os.listdir('data/images_full'):
    if f.endswith('.jpg') and f.replace('.jpg','') not in allowed:
        os.remove(os.path.join('data/images_full', f))
        removed += 1
print(f'Deleted {removed} images, {len(allowed)} remaining')
"
```

### 8. Restart the service

```bash
systemctl restart millenium-eye
```

The server now runs ORB with the correct Edison card set while the CNN
training happens on your Mac.

### 9. After Mac training is complete and ONNX committed

```bash
git pull
.venv/bin/pip install onnxruntime
rm -f data/cnn_embeddings.npy data/cnn_card_ids.npy data/cnn_card_names.json
.venv/bin/python3 build_database.py
systemctl restart millenium-eye
```

build_database.py will detect data/card_embeddings.onnx and build the
CNN index instead of ORB.
