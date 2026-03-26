# Training Steps

Run these in order from the project root on your Mac.

## 1. Revert images dir (already done if rsync used images_full)

Make sure generate_training_data.py points to data/images_full (not data/images).
It should already be correct in the repo.

## 2. Generate synthetic training data (~1-2 hours)

```bash
.venv/bin/python3 generate_training_data.py
```

Produces ~229,000 augmented images in data/synthetic/.

## 3. Fine-tune the model on M1 (~1-2 hours)

```bash
.venv/bin/python3 train_model.py
```

Saves weights to data/finetuned_mobilenet.pth.
Watch the accuracy climb each epoch — expect 0.7+ by the end.

## 4. Export to ONNX

```bash
.venv/bin/python3 export_onnx.py
```

Produces data/card_embeddings.onnx (~10 MB).

## 5. Build the embedding index

```bash
.venv/bin/python3 build_database.py
```

Produces data/cnn_embeddings.npy, data/cnn_card_ids.npy, data/cnn_card_names.json.

## 6. Commit and push

```bash
git add data/card_embeddings.onnx data/class_to_card_id.json data/cnn_card_names.json data/cnn_card_ids.npy data/cnn_embeddings.npy
git commit -m "Add fine-tuned card embedding model and index"
git push
```

## 7. Deploy to server

```bash
ssh millenium-eye
cd /opt/millenium-eye
git pull
.venv/bin/pip install onnxruntime
systemctl restart millenium-eye
```

## 8. Revert generate_training_data.py if needed

If generate_training_data.py was pointing to data/images instead of
data/images_full, fix it before committing:

In generate_training_data.py line 15:
    IMAGES_DIR = os.path.join("data", "images_full")
