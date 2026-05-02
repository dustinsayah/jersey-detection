"""Generates colab/clipt_action_model.ipynb.

v8.34.6 update — reads training_manifest.json (the comment-aware label set
from clipt-test/build_training_manifest.py), works with the labeling_clips/
MP4 directory, supports binary CUT-vs-keep training when GREAT+GOOD < 30
(the current state) and 3-class training when ≥ 30.
"""
import json
from pathlib import Path

NB = Path(__file__).parent / "clipt_action_model.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in text.strip().split("\n")]}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in src.rstrip("\n").split("\n")]}


CELLS = [
    md("""# Clipt — X3D-XS QB Highlight Classifier (Colab)

Fine-tune **X3D-XS** (3.79M params) on Dustin's labels → export INT8 ONNX → drop into Railway.

**Inputs you provide on Drive at `MyDrive/Clipt/`:**
- `training_manifest.json` (from `build_training_manifest.py`)
- `labeling_clips/` directory of MP4s (~1-3MB each, 854x480)

**Outputs (saved back to `MyDrive/Clipt/models/`):**
- `qb_x3d_xs_int8.onnx` — INT8 quantized, ~5MB, Railway-ready
- training metrics + confusion matrix

**Recommended runtime:** GPU (T4 minimum; A100 finishes in ~10 min)."""),

    md("## Cell 1 — GPU check"),
    code("""import torch
assert torch.cuda.is_available(), 'Need GPU runtime: Runtime → Change runtime type → GPU'
print(f'GPU:   {torch.cuda.get_device_name(0)}')
print(f'VRAM:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'Torch: {torch.__version__}')"""),

    md("## Cell 2 — Install"),
    code("""!pip install -q pytorchvideo onnx onnxruntime opencv-python-headless scikit-learn"""),

    md("## Cell 3 — Mount Drive + paths"),
    code("""from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE = '/content/drive/MyDrive/Clipt'
CLIPS_DIR = f'{DRIVE}/labeling_clips'
MANIFEST = f'{DRIVE}/training_manifest.json'
OUT_DIR = f'{DRIVE}/models'
os.makedirs(OUT_DIR, exist_ok=True)

assert os.path.exists(MANIFEST), f'Missing {MANIFEST} — upload training_manifest.json to Drive'
assert os.path.isdir(CLIPS_DIR), f'Missing {CLIPS_DIR} — upload labeling_clips folder'
print('manifest:', os.path.getsize(MANIFEST), 'bytes')
print('clips dir:', len([f for f in os.listdir(CLIPS_DIR) if f.endswith(\".mp4\")]), 'mp4 files')"""),

    md("""## Cell 4 — Parse manifest + decide training mode

If GREAT+GOOD ≥ 30 → 3-class (CUT/GOOD/GREAT). Otherwise → binary (CUT vs keep)
because there isn't enough signal to separate GOOD from GREAT."""),
    code("""import json
with open(MANIFEST) as f:
    bundle = json.load(f)

samples = bundle.get('samples', bundle if isinstance(bundle, list) else [])
distribution = bundle.get('distribution', {})
print(f'Samples: {len(samples)}')
print(f'Distribution: {distribution}')

great_good = sum(1 for s in samples if s['final_label'] in ('GREAT', 'GOOD'))
TRAINING_MODE = '3class' if great_good >= 30 else 'binary'
print(f'\\nTRAINING MODE: {TRAINING_MODE}  (GREAT+GOOD = {great_good})')

if TRAINING_MODE == 'binary':
    NUM_CLASSES = 2
    # 0=CUT (don't include), 1=keep (GOOD or GREAT)
    def remap(s):
        return 0 if s['final_label'] == 'CUT' else 1
else:
    NUM_CLASSES = 3
    def remap(s):
        return s['class']  # already 0=CUT, 1=GOOD, 2=GREAT

# Sanity check video files exist
import os
missing = []
for s in samples:
    cid = s['id']
    found = [f for f in os.listdir(CLIPS_DIR) if f.startswith(cid + '_')]
    if not found:
        missing.append(cid)
        s['_video_path'] = None
    else:
        s['_video_path'] = os.path.join(CLIPS_DIR, found[0])
        s['_class'] = remap(s)

print(f'\\nMissing videos: {len(missing)}')
if missing[:5]:
    print('  example:', missing[:5])
samples = [s for s in samples if s.get('_video_path')]
print(f'Usable samples: {len(samples)}')"""),

    md("""## Cell 5 — 16-frame clip extractor

X3D-XS input is 13×182×182 (channels first). We extract 16 evenly-spaced
frames across the labeled clip span, center-crop to square, resize to 182."""),
    code("""import cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

T_FRAMES = 16
SIZE = 182

def load_clip_frames(video_path: str, T: int = T_FRAMES, size: int = SIZE) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n_frames < 2:
        cap.release()
        return np.zeros((T, size, size, 3), dtype=np.uint8)
    indices = np.linspace(0, n_frames - 1, T, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frames.append(np.zeros((size, size, 3), dtype=np.uint8))
            continue
        h, w = frame.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = frame[y0:y0+side, x0:x0+side]
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0)

# Test
demo = load_clip_frames(samples[0]['_video_path'])
print(f'demo clip frames: {demo.shape} dtype={demo.dtype}')"""),

    md("## Cell 6 — Dataset + class weights"),
    code("""class ClipDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        arr = load_clip_frames(s['_video_path'])
        x = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 255.0  # (C,T,H,W)
        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        std  = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
        x = (x - mean) / std
        if self.augment and np.random.rand() < 0.5:
            x = torch.flip(x, dims=[-1])  # horizontal flip
        return x, s['_class']

# Stratified split
from sklearn.model_selection import train_test_split
labels = [s['_class'] for s in samples]
strat = labels if len(set(labels)) > 1 else None
train_s, val_s = train_test_split(samples, test_size=0.20, stratify=strat, random_state=42)
print(f'train: {len(train_s)} | val: {len(val_s)}')

# Class weights — strong because of imbalance
import numpy as np
counts = np.bincount([s['_class'] for s in train_s], minlength=NUM_CLASSES).astype(float)
weights = (1.0 / np.maximum(counts, 1.0))
weights = weights / weights.sum() * NUM_CLASSES
print(f'class counts: {counts.tolist()}')
print(f'class weights: {weights.tolist()}')

train_dl = DataLoader(ClipDataset(train_s, augment=True),  batch_size=4, shuffle=True,  num_workers=2)
val_dl   = DataLoader(ClipDataset(val_s,   augment=False), batch_size=4, shuffle=False, num_workers=2)"""),

    md("## Cell 7 — Load X3D-XS, swap head, fine-tune"),
    code("""import torch, torch.nn as nn

model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=True)
in_features = model.blocks[5].proj.in_features
model.blocks[5].proj = nn.Linear(in_features, NUM_CLASSES)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

w_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=w_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
EPOCHS = 25
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


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
        torch.save(model.state_dict(), f'{OUT_DIR}/qb_x3d_xs.pth')
    print(f'ep {ep+1:02d} | train loss {tr_loss:.3f} acc {tr_acc:.2%} | val loss {va_loss:.3f} acc {va_acc:.2%}{flag}')
print(f'\\nbest val acc: {best_val:.2%}')"""),

    md("## Cell 8 — Confusion matrix on validation set"),
    code("""import torch, numpy as np
model.load_state_dict(torch.load(f'{OUT_DIR}/qb_x3d_xs.pth'))
model.eval()

ys, preds = [], []
with torch.no_grad():
    for x, y in val_dl:
        x = x.to(device)
        out = model(x).argmax(1).cpu().numpy()
        ys.extend(y.numpy().tolist())
        preds.extend(out.tolist())

import collections
class_names = ['CUT', 'KEEP'] if NUM_CLASSES == 2 else ['CUT', 'GOOD', 'GREAT']
print('=== CONFUSION MATRIX ===')
print(' ' * 8 + 'pred:  ' + '  '.join(f'{n:>5s}' for n in class_names))
for i, n in enumerate(class_names):
    row = [sum(1 for y, p in zip(ys, preds) if y == i and p == j) for j in range(NUM_CLASSES)]
    print(f'true {n:5s}      ' + '  '.join(f'{x:>5d}' for x in row))

acc_per_class = []
for i, n in enumerate(class_names):
    truth = [1 if y == i else 0 for y in ys]
    pred  = [1 if p == i else 0 for p in preds]
    if sum(truth) == 0:
        print(f'{n}: no examples in val')
        continue
    correct_pos = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 1)
    print(f'{n}: recall {correct_pos / max(sum(truth), 1):.1%}, '
          f'precision {correct_pos / max(sum(pred), 1):.1%}')"""),

    md("## Cell 9 — ONNX export + INT8 quantize"),
    code("""import torch, os
model.load_state_dict(torch.load(f'{OUT_DIR}/qb_x3d_xs.pth'))
model.eval().cpu()
dummy = torch.randn(1, 3, T_FRAMES, SIZE, SIZE)
fp32 = f'{OUT_DIR}/qb_x3d_xs_fp32.onnx'
torch.onnx.export(
    model, dummy, fp32,
    input_names=['video'], output_names=['logits'],
    dynamic_axes={'video': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17,
)
print('FP32 ONNX:', os.path.getsize(fp32) // 1024, 'KB')

from onnxruntime.quantization import quantize_dynamic, QuantType
int8 = f'{OUT_DIR}/qb_x3d_xs_int8.onnx'
quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
print('INT8 ONNX:', os.path.getsize(int8) // 1024, 'KB')"""),

    md("## Cell 10 — CPU inference benchmark"),
    code("""import onnxruntime as ort, time, numpy as np
sess = ort.InferenceSession(f'{OUT_DIR}/qb_x3d_xs_int8.onnx',
                            providers=['CPUExecutionProvider'])
x = np.random.randn(1, 3, T_FRAMES, SIZE, SIZE).astype(np.float32)
for _ in range(3):
    sess.run(None, {'video': x})
t0 = time.time()
for _ in range(10):
    out = sess.run(None, {'video': x})
ms = (time.time() - t0) * 100
print(f'CPU inference: {ms:.0f} ms / clip')
if ms < 5000:
    print('Within Railway 5s budget')
else:
    print('Slower than 5s budget — may need x3d_xs or further quantization')"""),

    md("## Cell 11 — Wiring instructions"),
    code("""print(f'''
Model file: {OUT_DIR}/qb_x3d_xs_int8.onnx

NEXT STEPS:
1. Download {OUT_DIR}/qb_x3d_xs_int8.onnx from Drive to your local machine.
2. Upload to Cloudinary as raw resource:
     curl -X POST https://api.cloudinary.com/v1_1/dc33vjyyv/raw/upload \\
       -F 'file=@qb_x3d_xs_int8.onnx' \\
       -F 'upload_preset=clipt_uploads'
   Copy the secure_url from the response.
3. In Railway dashboard → jersey-detection service → Variables, add:
     QB_ACTION_ONNX_URL = <the secure_url from step 2>
4. Railway redeploys automatically (~1h15m).
5. Next detection request will fetch the ONNX, cache at /tmp, and use it.
6. Verify: clips will have a new "actionScore" field in their positionScore.
''')"""),
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

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote {NB} ({NB.stat().st_size // 1024} KB, {len(CELLS)} cells)")
