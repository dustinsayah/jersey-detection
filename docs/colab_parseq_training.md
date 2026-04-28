# Colab Notebook Design — PARSeq Jersey OCR Fine-Tuning

**Status:** design / research — not yet implemented as `.ipynb`.

## Goal

Replace EasyOCR (currently ~8% jersey hit rate at 360p) with a PARSeq model
fine-tuned on jersey crops from Dustin's actual game film. Target: 70–85%
hit rate per the Koshkina CVPR'24 paper for hockey/SoccerNet.

## Key research findings driving the design

- The Koshkina pipeline at <https://github.com/mkoshkina/jersey-number-pipeline>
  ships **pretrained PARSeq weights for hockey and SoccerNet** plus a legibility
  classifier. Both are downloaded into `models/`.
- **Data format:** weakly-labelled jersey crops in **LMDB**. The pipeline
  includes a script that takes a folder of crops + a labels file and packs
  them into the LMDB layout PARSeq expects.
- **Training command:** `python3 main.py SoccerNet train --train_str` (the
  paper uses SoccerNet as the example dataset; the same flow accepts a custom
  dataset name).
- **Minimum dataset size:** the upstream PARSeq repo doesn't publish a hard
  minimum, but practical fine-tuning of STR models converges with **500–1,000
  labeled examples**. Plan for at least 500 to start.

## Notebook outline (cells)

### Cell 1 — Setup

```python
from google.colab import drive
drive.mount('/content/drive')

import torch
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

```bash
%%bash
git clone https://github.com/mkoshkina/jersey-number-pipeline.git /content/jpipe
cd /content/jpipe
pip install -q -r requirements.txt
# PARSeq deps
pip install -q timm pytorch-lightning lmdb hydra-core
mkdir -p /content/jpipe/models
# Pull pretrained weights (URLs documented in the repo README; user should
# verify the latest links before running)
# wget -P /content/jpipe/models <PARSeq_HOCKEY_URL>
# wget -P /content/jpipe/models <LEGIBILITY_HOCKEY_URL>
```

### Cell 2 — Generate training crops from Dustin's game

This is where the v8.32.0 detection output becomes the data source. For every
moment with `target_visible=True` we extract the player crop that produced
the OCR hit (or the highest-scoring detection in the frame).

```python
import cv2, json, pathlib, os
SRC = '/content/drive/MyDrive/clipt/source.mp4'   # cached game film
DETECTION = json.load(open('/content/drive/MyDrive/clipt/detection_v832.json'))

OUT_IMG = pathlib.Path('/content/data/crops'); OUT_IMG.mkdir(parents=True, exist_ok=True)
labels = []     # list of (filename, jersey_number)

cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS)
TARGET = '2'    # Dustin's jersey

# Sample 5 frames per clip; persist crops at 2x for OCR-friendly input
for clip in DETECTION['clips']:
    if not clip.get('jerseyDetected'):
        continue
    for i, t in enumerate([
        clip['startTime'] + (clip['endTime']-clip['startTime']) * f
        for f in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]):
        cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
        ret, frame = cap.read()
        if not ret:
            continue
        # TODO: re-run YOLO+ByteTrack here to find the crop bbox. For the
        # first iteration we can save the full frame and hand-label.
        fn = f"clip{clip.get('peakTimestamp', 0):.0f}_f{i}.jpg"
        cv2.imwrite(str(OUT_IMG / fn), frame)
        labels.append((fn, TARGET))

# Plus negative examples: pull frames that flagged OTHER jersey numbers from
# the v8.32.0 track_jerseys log (not yet exported — needs a follow-up flag in
# bytetrack_pipeline.py to dump per-track OCR votes).
print('Saved', len(labels), 'crops')
pathlib.Path('/content/data/labels.txt').write_text(
    '\n'.join(f'{fn} {num}' for fn, num in labels)
)
```

**Manual labeling pass (required for first iteration):** open
`/content/data/crops/` and reject frames where the jersey isn't visible.
Aim for ~500 confirmed crops covering at least 5–10 unique jersey numbers
(Dustin's #2 plus opposing team numbers from the sideline).

### Cell 3 — Pack into LMDB

The Koshkina repo includes a `tools/create_lmdb_dataset.py` (matches the
upstream PARSeq repo's helper). Path may be different; verify before running.

```bash
%%bash
cd /content/jpipe
python tools/create_lmdb_dataset.py \
    --inputPath /content/data/crops \
    --gtFile /content/data/labels.txt \
    --outputPath /content/data/lmdb_train
# Mirror: small validation split
python tools/create_lmdb_dataset.py \
    --inputPath /content/data/crops_val \
    --gtFile /content/data/labels_val.txt \
    --outputPath /content/data/lmdb_val
```

### Cell 4 — Fine-tune PARSeq

The pipeline's wrapper script:

```bash
%%bash
cd /content/jpipe
# The README's example fine-tunes from SoccerNet weights. For our case start
# from the hockey weights (closer domain — broadcast sports, not soccer field
# top-down).
python3 main.py Clipt train --train_str \
    --pretrained_str models/parseq_hockey.ckpt \
    --train_lmdb /content/data/lmdb_train \
    --val_lmdb /content/data/lmdb_val \
    --max_epochs 10 \
    --batch_size 64
```

**Expected timing on A100:** ~2 minutes/epoch on 500–1,000 crops. 10 epochs ≈
20 minutes. Watch for validation accuracy to plateau; 10 may be more than
needed.

### Cell 5 — Evaluate + export

```python
# Run inference on a held-out validation set; report top-1 accuracy
# (jersey number exact match) and top-1 with edit distance ≤ 1.
# Then export the checkpoint.
import shutil
shutil.copy('/content/jpipe/runs/Clipt/last.ckpt',
            '/content/drive/MyDrive/clipt/parseq_clipt.ckpt')
```

### Cell 6 — Wire into Railway

Two integration points:

1. **Replace EasyOCR call** in `app/services/bytetrack_pipeline.py`'s
   `read_jersey_number`. PARSeq inference is heavier than EasyOCR but more
   accurate — accept the speed cost only after validation.
2. **Cache weights in Docker.** Add a download step to the Dockerfile so the
   image bakes in the fine-tuned `.ckpt` instead of pulling at runtime. The
   existing pattern for YOLO weights (`app/model/`) is the template.

## Open questions before implementing

- Do we have **enough labeled jersey #2 crops** to fine-tune meaningfully?
  v8.32.0's first run will tell us; if the answer is <100 visible-jersey
  moments, we need either upscaled input first OR synthetic data.
- Hockey-pretrained or SoccerNet-pretrained as the starting point? Hockey is
  closer to broadcast-sports text style; SoccerNet has more diverse angles.
  Start with hockey, swap if validation stalls.
- Should we co-train with one of the Roboflow Universe American football
  jersey datasets (e.g. **Football Jersey Tracker**, ~2,918 images)? Likely
  yes for the first round to compensate for our small in-game corpus.

## Roboflow datasets to pull for augmentation

- `football-tracking/football-jersey-tracker` — 2,918 images
- `cbu/jersey-number-detection` — 111 images (small but on-domain)
- `yakovk/jersey-numbers-i1wn5` — 556 images (mixed sports)

These give us a base of ~3,500 labeled jersey-number crops *before* adding any
of Dustin's footage. Pretraining on these then fine-tuning on Dustin-specific
crops is the standard recipe and the most robust path to the 70–85% hit-rate
target.
