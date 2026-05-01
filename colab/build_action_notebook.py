"""Generates colab/clipt_action_model.ipynb for fine-tuning X3D-S on labeled QB clips.

Run this script locally to (re)build the notebook. The notebook itself is the
artifact users open in Colab.

Why X3D-S (per Phase 0 research):
  - 3.79M params; smallest 3D-conv model with K-400 Top-1 ≥ 73%
  - Input 13x182, fits 360p HUDL footage natively
  - PyTorchVideo loader: torch.hub.load('facebookresearch/pytorchvideo','x3d_s')
  - ONNX exportable; with INT8 dynamic quantization runs ~1.5-3s on Railway CPU
  - Tiny enough to fine-tune on ~200 labeled clips without overfitting
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "clipt_action_model.ipynb"


def cell_md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def cell_code(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.rstrip("\n").split("\n")],
    }


CELLS = [
    cell_md("""# Clipt — X3D-S QB Highlight Classifier (Colab)

Fine-tune **X3D-S** (3.79M params) on labeled QB clips → export INT8 ONNX → drop into Railway pipeline.

**Inputs you provide:**
- `labeling_results.json` (from `labeling_tool.html`)
- `source.mp4` (the game film)

**Outputs:**
- `qb_x3d_s.pth` (PyTorch checkpoint)
- `qb_x3d_s_int8.onnx` (Railway-ready, ~5MB)
- Eval metrics (held-out accuracy)

**Recommended runtime:** GPU (T4 or A100). CPU works for inference but training will be slow."""),

    cell_md("## Cell 1 — GPU check"),
    cell_code("""import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print('PyTorch:', torch.__version__)"""),

    cell_md("## Cell 2 — Install deps"),
    cell_code("""# pytorchvideo loads X3D from torch hub. Need a few helpers.
!pip install -q pytorchvideo onnx onnxruntime opencv-python einops
import os, sys
print('python', sys.version.split()[0])"""),

    cell_md("""## Cell 3 — Mount Drive and configure paths

Put `labeling_results.json` and `source.mp4` in your Drive at the paths below
(or change the constants to wherever they live)."""),
    cell_code("""from google.colab import drive
drive.mount('/content/drive')

import os
WORK = '/content/clipt_action'
os.makedirs(WORK, exist_ok=True)

# === EDIT THESE PATHS ===
LABELS_JSON = '/content/drive/MyDrive/Clipt/labeling_results.json'
SOURCE_MP4  = '/content/drive/MyDrive/Clipt/source.mp4'
# ========================

assert os.path.exists(LABELS_JSON), f'missing {LABELS_JSON}'
assert os.path.exists(SOURCE_MP4),  f'missing {SOURCE_MP4}'
print('labels:', os.path.getsize(LABELS_JSON), 'bytes')
print('source:', os.path.getsize(SOURCE_MP4) // 1024 // 1024, 'MB')"""),

    cell_md("""## Cell 4 — Parse labels

Convert GREAT/GOOD/CUT into a 3-class problem. SKIP entries are dropped."""),
    cell_code("""import json
with open(LABELS_JSON) as f:
    data = json.load(f)

raw = data.get('labels', [])
LABEL_TO_IDX = {'GREAT': 2, 'GOOD': 1, 'CUT': 0}

samples = []
for item in raw:
    lbl = item.get('label')
    if lbl not in LABEL_TO_IDX:
        continue
    samples.append({
        'idx': item['idx'],
        'start': float(item['startTime']),
        'end':   float(item['endTime']),
        'label': LABEL_TO_IDX[lbl],
        'label_str': lbl,
    })

print(f'{len(samples)} usable samples')
from collections import Counter
print(Counter(s['label_str'] for s in samples))
assert len(samples) >= 30, 'Need at least ~30 labeled clips before fine-tuning is meaningful.'"""),

    cell_md("""## Cell 5 — Extract 16-frame clips at 2 fps

X3D-S input is 13×182×182. We extract 16 frames sampled across the clip
duration, resize to 182×182, and stack as a (T, H, W, C) tensor."""),
    cell_code("""import cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

T_FRAMES = 16          # number of frames per clip
SIZE     = 182         # X3D-S spatial input

def load_clip(path, start_s, end_s, T=T_FRAMES, size=SIZE):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = max(end_s - start_s, 0.5)
    times = np.linspace(start_s, end_s, T)
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            frames.append(np.zeros((size, size, 3), dtype=np.uint8))
            continue
        # Center-crop to square then resize.
        h, w = frame.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        frame = frame[y0:y0+side, x0:x0+side]
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    arr = np.stack(frames, axis=0)  # (T, H, W, 3) uint8
    return arr

class ClipDS(Dataset):
    def __init__(self, samples, source, augment=False):
        self.samples = samples
        self.source = source
        self.augment = augment
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        arr = load_clip(self.source, s['start'], s['end'])
        # (T,H,W,C) -> (C,T,H,W) for X3D
        x = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 255.0
        # Standard normalization (Kinetics stats)
        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        std  = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
        x = (x - mean) / std
        if self.augment and np.random.rand() < 0.5:
            x = torch.flip(x, dims=[-1])  # horizontal flip
        return x, s['label']

# Sanity: pull one clip
demo = ClipDS(samples[:1], SOURCE_MP4)[0]
print('clip tensor:', demo[0].shape, 'label:', demo[1])"""),

    cell_md("""## Cell 6 — Train/eval split & DataLoader"""),
    cell_code("""from sklearn.model_selection import train_test_split
import torch

train_s, val_s = train_test_split(
    samples, test_size=0.2,
    stratify=[s['label'] for s in samples] if len(set(s['label'] for s in samples)) > 1 else None,
    random_state=42,
)
print('train:', len(train_s), 'val:', len(val_s))

train_ds = ClipDS(train_s, SOURCE_MP4, augment=True)
val_ds   = ClipDS(val_s,   SOURCE_MP4, augment=False)

# Small batch for X3D-S on T4 (it's tight on VRAM with T=16, S=182)
train_dl = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=2)
val_dl   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=2)"""),

    cell_md("""## Cell 7 — Load X3D-S, replace classifier head, fine-tune"""),
    cell_code("""import torch, torch.nn as nn

# Load X3D-S pretrained on Kinetics-400
model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
# Replace 400-way K-400 head with 3-way (CUT/GOOD/GREAT)
in_features = model.blocks[5].proj.in_features
model.blocks[5].proj = nn.Linear(in_features, 3)
model = model.cuda() if torch.cuda.is_available() else model

# Class weights — labels are usually imbalanced (lots of CUT, few GREAT).
import numpy as np
counts = np.bincount([s['label'] for s in train_s], minlength=3).astype(float)
weights = (1.0 / np.maximum(counts, 1.0))
weights = torch.tensor(weights / weights.sum() * len(weights), dtype=torch.float32)
print('class weights (CUT/GOOD/GREAT):', weights.tolist())

criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights.to(next(model.parameters()).device))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

EPOCHS = 20
device = next(model.parameters()).device

def run_epoch(dl, train=True):
    model.train() if train else model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct  += (out.argmax(1) == y).sum().item()
            tot      += x.size(0)
    return loss_sum / tot, correct / tot

best_val = 0.0
for ep in range(EPOCHS):
    tr_loss, tr_acc = run_epoch(train_dl, train=True)
    va_loss, va_acc = run_epoch(val_dl, train=False)
    scheduler.step()
    flag = ' <- best' if va_acc > best_val else ''
    if va_acc > best_val:
        best_val = va_acc
        torch.save(model.state_dict(), f'{WORK}/qb_x3d_s.pth')
    print(f'ep {ep+1:02d}  tr_loss {tr_loss:.3f} tr_acc {tr_acc:.2%}  val_loss {va_loss:.3f} val_acc {va_acc:.2%}{flag}')

print(f'\\nbest val acc: {best_val:.2%}')"""),

    cell_md("""## Cell 8 — ONNX export with INT8 dynamic quantization

ONNX Runtime + INT8 brings X3D-S CPU inference into the ~1.5-3s/clip range
on a Railway shared vCPU (vs. 5-10s+ in PyTorch FP32)."""),
    cell_code("""import torch, os
model.load_state_dict(torch.load(f'{WORK}/qb_x3d_s.pth'))
model.eval().cpu()

# Trace with dummy input
dummy = torch.randn(1, 3, T_FRAMES, SIZE, SIZE)
onnx_fp32 = f'{WORK}/qb_x3d_s_fp32.onnx'
torch.onnx.export(
    model, dummy, onnx_fp32,
    input_names=['video'], output_names=['logits'],
    dynamic_axes={'video': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17,
)
print('FP32 ONNX:', os.path.getsize(onnx_fp32) // 1024, 'KB')

from onnxruntime.quantization import quantize_dynamic, QuantType
onnx_int8 = f'{WORK}/qb_x3d_s_int8.onnx'
quantize_dynamic(onnx_fp32, onnx_int8, weight_type=QuantType.QInt8)
print('INT8 ONNX:', os.path.getsize(onnx_int8) // 1024, 'KB')"""),

    cell_md("""## Cell 9 — CPU benchmark on the INT8 model"""),
    cell_code("""import onnxruntime as ort, time, numpy as np

sess = ort.InferenceSession(onnx_int8, providers=['CPUExecutionProvider'])
x = np.random.randn(1, 3, T_FRAMES, SIZE, SIZE).astype(np.float32)

# warmup
for _ in range(3):
    sess.run(None, {'video': x})

n = 5
t0 = time.time()
for _ in range(n):
    out = sess.run(None, {'video': x})
elapsed = (time.time() - t0) / n
print(f'CPU inference: {elapsed*1000:.0f} ms / clip')
print('logits:', out[0])"""),

    cell_md("""## Cell 10 — Save outputs to Drive + wiring instructions"""),
    cell_code("""import shutil
DRIVE_OUT = '/content/drive/MyDrive/Clipt/models'
os.makedirs(DRIVE_OUT, exist_ok=True)
shutil.copy(f'{WORK}/qb_x3d_s.pth',     f'{DRIVE_OUT}/qb_x3d_s.pth')
shutil.copy(f'{WORK}/qb_x3d_s_int8.onnx', f'{DRIVE_OUT}/qb_x3d_s_int8.onnx')

print('Saved to:', DRIVE_OUT)
print()
print('TO WIRE INTO RAILWAY DETECTION:')
print('  1. Upload qb_x3d_s_int8.onnx somewhere Railway can fetch (Cloudinary, S3, raw.githubusercontent).')
print('  2. Set env var QB_ACTION_ONNX_URL=<that url> in the Railway service.')
print('  3. The detection server (when app/services/action_scorer.py lands) will:')
print('       - Lazy-fetch and cache the ONNX file on first run')
print('       - For each candidate clip, extract 16 frames around peak_time')
print('       - Run inference, append `actionScore` to the clip dict')
print('       - position_scorer.score_qb_clip already accepts an optional')
print('         action_score dimension; weight it 25-30% of final score.')"""),

    cell_md("""## Cell 11 — Notes on iterating

If validation accuracy plateaus < 70%:
- Add more labels (target ≥150 per class ideally)
- Try X3D-M instead of X3D-S (4× more compute, +3 K-400 points)
- Try VideoMAE-Small as the backbone (HF: MCG-NJU/videomae-small-finetuned-kinetics)

If you want to label clips from MULTIPLE games:
- Run `extract_labeling_frames.py` on each game, label, then merge JSONs
- More variety > more from the same game

For RB / WR / LB rubrics later:
- Same notebook, change LABEL_TO_IDX semantics + relabel
- Keep ONE model per position, OR train a multi-position model and conditionally
  weight outputs in `position_scorer.py`."""),
]


nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote {NB_PATH} ({NB_PATH.stat().st_size // 1024} KB, {len(CELLS)} cells)")
