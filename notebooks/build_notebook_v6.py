#!/usr/bin/env python3
"""Build the v6 training notebook programmatically."""
import json
import sys

cells = []


def md(source):
    lines = source.split("\n")
    cells.append({
        "cell_type": "markdown", "metadata": {},
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
    })


def code(source):
    lines = source.split("\n")
    cells.append({
        "cell_type": "code", "metadata": {},
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
        "outputs": [], "execution_count": None,
    })


# ═══════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════
md("""# Clipt v6 \u2014 Final Training Pipeline

**THE LAST NOTEBOOK.** Trains every missing model for the Clipt detection pipeline.

| Section | Model | Type | Output |
|---------|-------|------|--------|
| 1 | Scoreboard Detector v5 | YOLO detect | `scoreboard_detector_v5.pt` |
| 2 | Dead Ball Classifier v5 | YOLO classify | `dead_ball_classifier_v5.pt` |
| 3 | Jersey Super-Resolution v5 | PyTorch SR (SRResNet) | `jersey_upscaler_v5.pth` |
| 4 | Player Isolator v3 | YOLO detect | `player_isolator_v3.pt` |
| 5 | Basketball Jersey OCR v3 | YOLO detect | `basketball_jersey_number_v3.pt` |
| 6 | Basketball Player Detector v2 | YOLO detect | `basketball_player_detector_v2.pt` |
| 7 | Lacrosse Detector v2 | YOLO detect | `lacrosse_detector_v2.pt` |

**After training:** copy all files from Google Drive \u2192 `reelapp/playerJerseyIdentification-master/app/model/`

**Estimated time:** ~3\u20134 hours on A100 (all 7 models)
**Estimated cost:** ~$2\u20135 Anthropic API for auto-labeling""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 0: SETUP
# ═══════════════════════════════════════════════════════════════════
md("---\n## Section 0: Setup")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 0A \u2014 Install Packages
# ═══════════════════════════════════════════════════════
!pip install roboflow ultralytics pyyaml anthropic yt-dlp pillow -q
print("\\n\u2705 All packages installed")""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 0B \u2014 Imports + Config
# ═══════════════════════════════════════════════════════
import os, sys, torch, shutil, glob, yaml, time, json, cv2, gc, zipfile, subprocess, random
import numpy as np
from collections import Counter
from pathlib import Path
from google.colab import userdata

# \u2500\u2500 Directories \u2500\u2500
DRIVE_SAVE_DIR = '/content/drive/MyDrive/clipt_v6_models'
DRIVE_CHECKPOINTS = '/content/drive/MyDrive/clipt_v6_models/checkpoints'
DATA_BASE = '/content/data'

# \u2500\u2500 API Keys \u2500\u2500
ROBOFLOW_API_KEY = ''
ANTHROPIC_API_KEY = ''
try:
    ROBOFLOW_API_KEY = userdata.get('ROBOFLOW_API_KEY')
except Exception:
    pass
try:
    ANTHROPIC_API_KEY = userdata.get('ANTHROPIC_API_KEY')
except Exception:
    pass

# \u2500\u2500 GPU Check \u2500\u2500
assert torch.cuda.is_available(), '\u274c NO GPU \u2014 Runtime > Change runtime type > A100'
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
HAS_A100 = vram_gb > 40
DEFAULT_BATCH = 16 if HAS_A100 else 8

# \u2500\u2500 Create Directories \u2500\u2500
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
os.makedirs(DRIVE_CHECKPOINTS, exist_ok=True)
os.makedirs(DATA_BASE, exist_ok=True)

print(f'\u2705 GPU: {gpu_name} ({vram_gb:.1f}GB)')
print(f'\u2705 Batch size: {DEFAULT_BATCH}')
print(f'\u2705 Roboflow API key: {"loaded" if ROBOFLOW_API_KEY else "\u274c MISSING"}')
print(f'\u2705 Anthropic API key: {"loaded" if ANTHROPIC_API_KEY else "\u274c MISSING (optional)"}')

# \u2500\u2500 Roboflow helper \u2500\u2500
def safe_download(ws, proj, ver, name, fmt='yolov8'):
    from roboflow import Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    for v in [ver] + [i for i in range(1, 8) if i != ver]:
        try:
            ds = rf.workspace(ws).project(proj).version(v).download(fmt)
            count = len(glob.glob(f'{ds.location}/train/images/*'))
            print(f'\u2705 {name} v{v}: {count} images')
            return ds
        except Exception as e:
            print(f'  v{v} failed: {e}')
    print(f'\u274c {name}: all versions failed')
    return None

# \u2500\u2500 YOLO save helper \u2500\u2500
def save_model_to_drive(model_name, task='detect'):
    paths = sorted(glob.glob(f'runs/{task}/{model_name}*/weights/best.pt'))
    if not paths:
        alt = f'{DRIVE_CHECKPOINTS}/{model_name}/best.pt'
        if os.path.exists(alt): paths = [alt]
    assert paths and os.path.exists(paths[-1]), f'\u274c No trained model \u2014 run training cell first'
    src = paths[-1]
    dst = f'{DRIVE_SAVE_DIR}/{model_name}.pt'
    shutil.copy2(src, dst)
    sz = os.path.getsize(dst) / 1024 / 1024
    print(f'{"=" * 60}')
    print(f'\u2705 {model_name}.pt SAVED TO DRIVE')
    print(f'Location: {DRIVE_SAVE_DIR}/')
    print(f'File size: {sz:.1f}MB')
    print(f'{"=" * 60}')
    return dst

def backup_checkpoints(model_name, task='detect'):
    ckpt_dir = f'{DRIVE_CHECKPOINTS}/{model_name}'
    os.makedirs(ckpt_dir, exist_ok=True)
    for wt in ['best.pt', 'last.pt']:
        paths = sorted(glob.glob(f'runs/{task}/{model_name}*/weights/{wt}'))
        if paths: shutil.copy2(paths[-1], f'{ckpt_dir}/{wt}')
    torch.cuda.empty_cache(); gc.collect()""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 0C \u2014 Mount Google Drive
# ═══════════════════════════════════════════════════════
from google.colab import drive
drive.mount('/content/drive')
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
os.makedirs(DRIVE_CHECKPOINTS, exist_ok=True)
print(f'\\n\u2705 Drive mounted')
print(f'\u2705 Models save to: {DRIVE_SAVE_DIR}')
print(f'   (copy finished files \u2192 reelapp/playerJerseyIdentification-master/app/model/)')
existing = glob.glob(f'{DRIVE_SAVE_DIR}/*.*')
if existing:
    print(f'\\nFound {len(existing)} existing files:')
    for f in sorted(existing):
        sz = os.path.getsize(f) / 1024 / 1024
        print(f'  {os.path.basename(f)} \u2014 {sz:.1f}MB')""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: SCOREBOARD DETECTOR v5
# ═══════════════════════════════════════════════════════════════════
md("""\
---
## Section 1: Scoreboard Detector v5
YOLO detect model \u2014 finds scoreboard/overlay regions in sports video frames.
Download scoreboard datasets from Roboflow + auto-label broadcast frames.
Output: `scoreboard_detector_v5.pt`""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 1A \u2014 Download scoreboard datasets
# ═══════════════════════════════════════════════════════
from roboflow import Roboflow
import yt_dlp

assert ROBOFLOW_API_KEY, '\u274c Set ROBOFLOW_API_KEY in Colab Secrets'

print('=== Downloading Scoreboard Detection Datasets ===')
ds1 = safe_download('aniruddha-mani', 'scoreboard-detection-tqacl', 1, 'scoreboard_det_1')
ds2 = safe_download('roboflow-universe-projects', 'scoreboard-detection-yb2aw', 1, 'scoreboard_det_2')
ds3 = safe_download('scoreboard', 'scoreboard-detector', 1, 'scoreboard_det_3')

# \u2500\u2500 Merge into single dataset \u2500\u2500
print('\\nMerging scoreboard datasets...')
MP = f'{DATA_BASE}/scoreboard_merged'
for s in ['train', 'val']:
    os.makedirs(f'{MP}/{s}/images', exist_ok=True)
    os.makedirs(f'{MP}/{s}/labels', exist_ok=True)

total_images = 0
for ds in [ds1, ds2, ds3]:
    if not ds: continue
    loc = ds.location; dn = os.path.basename(loc)
    for sp in ['train', 'valid', 'val', 'test']:
        idir = f'{loc}/{sp}/images'; ldir = f'{loc}/{sp}/labels'
        if not os.path.exists(idir): continue
        target = 'val' if sp in ['valid', 'val', 'test'] else 'train'
        for img_path in glob.glob(f'{idir}/*'):
            fn = f'{dn}_{os.path.basename(img_path)}'
            shutil.copy2(img_path, f'{MP}/{target}/images/{fn}')
            lbl = os.path.join(ldir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
            if os.path.exists(lbl):
                with open(lbl) as f:
                    lines = ['0 ' + ' '.join(l.strip().split()[1:]) for l in f if len(l.strip().split()) >= 5]
                if lines:
                    with open(f'{MP}/{target}/labels/{os.path.splitext(fn)[0]}.txt', 'w') as f:
                        f.write('\\n'.join(lines) + '\\n')
            total_images += 1

# \u2500\u2500 Extract frames from sports broadcasts for extra data \u2500\u2500
print('\\nExtracting broadcast frames for scoreboard data...')
frame_dir = f'{DATA_BASE}/scoreboard_frames'
os.makedirs(frame_dir, exist_ok=True)

QUERIES = [
    'basketball game broadcast scoreboard', 'football game broadcast scoreboard',
    'lacrosse game broadcast scoreboard', 'NBA game highlights',
    'NFL game highlights', 'college basketball game', 'college football broadcast',
]
for q in QUERIES:
    try:
        ydl_opts = {
            'format': 'best[height<=480]', 'outtmpl': f'{frame_dir}/%(id)s.%(ext)s',
            'quiet': True, 'max_downloads': 2,
            'extractor_args': {'youtube': {'player_client': ['android']}},
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'ytsearch2:{q}'])
    except Exception as e:
        print(f'  {q}: {e}')

for vf in glob.glob(f'{frame_dir}/*.mp4') + glob.glob(f'{frame_dir}/*.webm'):
    try:
        cap = cv2.VideoCapture(vf)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fi in range(0, total, max(1, total // 5)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret: continue
            fname = f'broadcast_{os.path.basename(vf)}_{fi}.jpg'
            cv2.imwrite(f'{MP}/train/images/{fname}', cv2.resize(frame, (640, 640)))
            with open(f'{MP}/train/labels/{fname.replace(".jpg",".txt")}', 'w') as f:
                f.write('0 0.5 0.075 0.95 0.15\\n')  # top-of-screen scoreboard
            total_images += 1
        cap.release()
    except Exception:
        pass
shutil.rmtree(frame_dir, ignore_errors=True)

dy = {'path': MP, 'train': 'train/images', 'val': 'val/images', 'nc': 1, 'names': {0: 'scoreboard'}}
with open(f'{MP}/data.yaml', 'w') as f:
    yaml.dump(dy, f)
tc = len(glob.glob(f'{MP}/train/images/*'))
vc = len(glob.glob(f'{MP}/val/images/*'))
print(f'\\n\u2705 Scoreboard dataset: {tc} train, {vc} val \u2014 1 class (scoreboard)')""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 1B \u2014 Train scoreboard_detector_v5
# ═══════════════════════════════════════════════════════
from ultralytics import YOLO

MODEL_NAME = 'scoreboard_detector_v5'
DATA_PATH = f'{DATA_BASE}/scoreboard_merged/data.yaml'

assert os.path.exists(DATA_PATH), '\u274c No scoreboard data \u2014 run Cell 1A first'
tc = len(glob.glob(f'{DATA_BASE}/scoreboard_merged/train/images/*'))
assert tc >= 30, f'\u274c Need 30+ images, have {tc}'
print(f'\u2705 {tc} training images ready')

print(f'\\n{"=" * 60}')
print(f'\U0001f680 TRAINING: {MODEL_NAME}')
print(f'   epochs=100, batch={DEFAULT_BATCH}, patience=20')
print(f'{"=" * 60}\\n')

resume_path = f'{DRIVE_CHECKPOINTS}/{MODEL_NAME}/last.pt'
if os.path.exists(resume_path) and os.path.getsize(resume_path) > 1024*1024:
    print('\u267b\ufe0f Resuming from Drive checkpoint...')
    model = YOLO(resume_path)
    model.train(resume=True)
else:
    model = YOLO('yolov8m.pt')
    model.train(data=DATA_PATH, epochs=100, imgsz=640, batch=DEFAULT_BATCH,
                name=MODEL_NAME, device=0, patience=20, save_period=5,
                amp=True, cache=True, mosaic=0.8, scale=0.3)

backup_checkpoints(MODEL_NAME, 'detect')
print(f'\\n\u2705 {MODEL_NAME} training complete!')""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 1C \u2014 Save scoreboard_detector_v5.pt to Drive
# ═══════════════════════════════════════════════════════
save_model_to_drive('scoreboard_detector_v5', 'detect')""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 2: DEAD BALL CLASSIFIER v5
# ═══════════════════════════════════════════════════════════════════
md("""\
---
## Section 2: Dead Ball Classifier v5
YOLO **classify** model \u2014 binary: `dead_ball` vs `live_play`.

Dead ball = timeouts, free throws, huddles, replays, celebrations, commercial breaks.
Live play = active game action, ball in motion, players running plays.

Uses Claude Haiku to auto-label extracted frames.
Output: `dead_ball_classifier_v5.pt`""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 2A \u2014 Harvest frames + auto-label dead ball
# ═══════════════════════════════════════════════════════
import yt_dlp, anthropic, base64

CLS_DIR = f'{DATA_BASE}/dead_ball_cls'
VID_DIR = f'{DATA_BASE}/dead_ball_videos'
os.makedirs(VID_DIR, exist_ok=True)

# Step 1: Download game footage
QUERIES = [
    'full basketball game highlights', 'full football game highlights',
    'full lacrosse game', 'basketball timeout free throw',
    'football huddle timeout', 'basketball halftime ceremony',
    'football replay review challenge', 'lacrosse timeout',
    'NBA game action plays', 'NFL game action plays',
    'college basketball game action', 'college football game action',
]

print('=== Downloading game footage for dead ball classification ===')
for q in QUERIES:
    try:
        ydl_opts = {
            'format': 'best[height<=480]', 'outtmpl': f'{VID_DIR}/%(id)s.%(ext)s',
            'quiet': True, 'max_downloads': 2,
            'extractor_args': {'youtube': {'player_client': ['android']}},
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'ytsearch2:{q}'])
        print(f'\u2705 {q}')
    except Exception as e:
        print(f'  \u274c {q}: {e}')

vid_files = glob.glob(f'{VID_DIR}/*.mp4') + glob.glob(f'{VID_DIR}/*.webm')
print(f'\\nDownloaded {len(vid_files)} videos')

# Step 2: Extract frames every 2 seconds
print('\\nExtracting frames...')
all_frames = []
os.makedirs(f'{DATA_BASE}/dead_ball_raw', exist_ok=True)
for vf in vid_files:
    cap = cv2.VideoCapture(vf)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(fps * 2))
    for fi in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: continue
        fname = f'{os.path.basename(vf)}_{fi}.jpg'
        fpath = f'{DATA_BASE}/dead_ball_raw/{fname}'
        cv2.imwrite(fpath, cv2.resize(frame, (640, 640)))
        all_frames.append(fpath)
    cap.release()
print(f'Extracted {len(all_frames)} frames')
if len(all_frames) > 2000:
    random.shuffle(all_frames)
    all_frames = all_frames[:2000]
    print(f'Limited to 2000 frames')

# Step 3: Claude auto-labeling
print('\\nAuto-labeling with Claude Haiku...')
assert ANTHROPIC_API_KEY, '\u274c Set ANTHROPIC_API_KEY in Colab Secrets'
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

for cls in ['dead_ball', 'live_play']:
    for split in ['train', 'val']:
        os.makedirs(f'{CLS_DIR}/{split}/{cls}', exist_ok=True)

labeled = 0; errors = 0; labels_cache = {}
for i, fpath in enumerate(all_frames):
    if labeled >= 1500: break
    try:
        with open(fpath, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        resp = client.messages.create(
            model='claude-haiku-4-5-20251001', max_tokens=20,
            messages=[{'role': 'user', 'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': img_b64}},
                {'type': 'text', 'text': 'Is this frame from a live sports play (players actively playing, ball in motion, game action) or a dead ball moment (timeout, free throw setup, huddle, replay, celebration, commercial, scoreboard-only, ceremony, halftime)? Reply LIVE or DEAD'},
            ]}]
        )
        answer = resp.content[0].text.strip().upper()
        label = 'dead_ball' if 'DEAD' in answer else 'live_play'
        split = 'val' if hash(fpath) % 5 == 0 else 'train'
        shutil.copy2(fpath, f'{CLS_DIR}/{split}/{label}/{os.path.basename(fpath)}')
        labels_cache[fpath] = label
        labeled += 1
        if labeled % 100 == 0:
            print(f'  Labeled {labeled}/{len(all_frames)}...')
            with open(f'{DRIVE_CHECKPOINTS}/dead_ball_labels.json', 'w') as f:
                json.dump(labels_cache, f)
    except Exception as e:
        errors += 1
        if errors > 20:
            print(f'\u274c Too many errors ({errors}), stopping')
            break

shutil.rmtree(f'{DATA_BASE}/dead_ball_raw', ignore_errors=True)
shutil.rmtree(VID_DIR, ignore_errors=True)
tc = len(glob.glob(f'{CLS_DIR}/train/**/*.jpg', recursive=True))
vc = len(glob.glob(f'{CLS_DIR}/val/**/*.jpg', recursive=True))
db_t = len(glob.glob(f'{CLS_DIR}/train/dead_ball/*'))
lp_t = len(glob.glob(f'{CLS_DIR}/train/live_play/*'))
print(f'\\n\u2705 Dead ball dataset: {tc} train ({db_t} dead, {lp_t} live), {vc} val')
print(f'   Labeled: {labeled}, Errors: {errors}')""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 2B \u2014 Train dead_ball_classifier_v5
# ═══════════════════════════════════════════════════════
from ultralytics import YOLO

MODEL_NAME = 'dead_ball_classifier_v5'
CLS_DATA = f'{DATA_BASE}/dead_ball_cls'
tc = len(glob.glob(f'{CLS_DATA}/train/**/*.jpg', recursive=True))
vc = len(glob.glob(f'{CLS_DATA}/val/**/*.jpg', recursive=True))
assert tc >= 50, f'\u274c Need 50+ train images, have {tc}'
print(f'\u2705 {tc} train, {vc} val images ready')

print(f'\\n{"=" * 60}')
print(f'\U0001f680 TRAINING: {MODEL_NAME}')
print(f'   epochs=80, YOLO classify mode (binary: dead_ball / live_play)')
print(f'{"=" * 60}\\n')

resume_path = f'{DRIVE_CHECKPOINTS}/{MODEL_NAME}/last.pt'
if os.path.exists(resume_path) and os.path.getsize(resume_path) > 1024*1024:
    print('\u267b\ufe0f Resuming from Drive checkpoint...')
    model = YOLO(resume_path)
    model.train(resume=True)
else:
    model = YOLO('yolov8m-cls.pt')
    model.train(data=CLS_DATA, epochs=80, imgsz=640, batch=DEFAULT_BATCH,
                name=MODEL_NAME, device=0, patience=15, save_period=5,
                amp=True, cache=True)

backup_checkpoints(MODEL_NAME, 'classify')
print(f'\\n\u2705 {MODEL_NAME} training complete!')""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 2C \u2014 Save dead_ball_classifier_v5.pt to Drive
# ═══════════════════════════════════════════════════════
save_model_to_drive('dead_ball_classifier_v5', 'classify')""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 3: JERSEY SUPER-RESOLUTION v5
# ═══════════════════════════════════════════════════════════════════
md("""\
---
## Section 3: Jersey Super-Resolution v5
**NOT a YOLO model.** Custom PyTorch SRResNet that 4x upscales small jersey crops before OCR.

Architecture: 8 residual blocks + 2x PixelShuffle (4x total upscale)
Training: paired low-res/high-res jersey crop images
Output: `jersey_upscaler_v5.pth`""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 3A \u2014 Build SR training dataset (paired LR/HR jersey crops)
# ═══════════════════════════════════════════════════════
from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont

assert ROBOFLOW_API_KEY, '\u274c Set ROBOFLOW_API_KEY in Colab Secrets'

SR_DIR = f'{DATA_BASE}/sr_dataset'
HR_DIR = f'{SR_DIR}/hr'   # high-res targets (128x128)
LR_DIR = f'{SR_DIR}/lr'   # low-res inputs (32x32)
os.makedirs(HR_DIR, exist_ok=True)
os.makedirs(LR_DIR, exist_ok=True)

# Step 1: Download jersey number datasets with bounding boxes
print('=== Downloading jersey datasets for SR training ===')
ds1 = safe_download('roboflow-universe-projects', 'jersey-number-detection-2', 1, 'jersey_numbers_1')
ds2 = safe_download('football-jersey', 'soccer-jersey-number', 1, 'jersey_numbers_2')
ds3 = safe_download('nfl-jersey-number', 'nfl-jersey', 1, 'jersey_numbers_3')

# Step 2: Extract jersey crops from annotated images
print('\\nExtracting jersey crops...')
crop_count = 0
for ds in [ds1, ds2, ds3]:
    if not ds: continue
    loc = ds.location
    for sp in ['train', 'valid', 'val', 'test']:
        idir = f'{loc}/{sp}/images'; ldir = f'{loc}/{sp}/labels'
        if not os.path.exists(idir): continue
        for img_path in glob.glob(f'{idir}/*'):
            try:
                img = cv2.imread(img_path)
                if img is None: continue
                ih, iw = img.shape[:2]
                lbl = os.path.join(ldir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
                if not os.path.exists(lbl): continue
                with open(lbl) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = max(0, int((cx - bw/2) * iw))
                        y1 = max(0, int((cy - bh/2) * ih))
                        x2 = min(iw, int((cx + bw/2) * iw))
                        y2 = min(ih, int((cy + bh/2) * ih))
                        if x2-x1 < 10 or y2-y1 < 10: continue
                        crop = img[y1:y2, x1:x2]
                        hr = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_CUBIC)
                        lr = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
                        fname = f'crop_{crop_count:05d}.png'
                        cv2.imwrite(f'{HR_DIR}/{fname}', hr)
                        cv2.imwrite(f'{LR_DIR}/{fname}', lr)
                        crop_count += 1
            except Exception:
                continue

# Step 3: Generate synthetic jersey number crops
print('\\nGenerating synthetic jersey crops...')
COLORS = [
    (255,255,255), (200,0,0), (0,0,200), (0,150,0), (255,165,0),
    (128,0,128), (0,0,0), (255,255,0), (0,200,200), (50,50,50),
]
for num in range(100):
    for ci, color in enumerate(COLORS):
        try:
            img = Image.new('RGB', (128, 128), color)
            draw = ImageDraw.Draw(img)
            text = str(num)
            try:
                font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 60)
            except Exception:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            fill = (255,255,255) if sum(color) < 400 else (0,0,0)
            draw.text(((128-tw)//2, (128-th)//2), text, fill=fill, font=font)
            hr_np = np.array(img)
            noise = np.random.randint(-15, 15, hr_np.shape, dtype=np.int16)
            hr_np = np.clip(hr_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            fname = f'synth_{num:02d}_c{ci}.png'
            cv2.imwrite(f'{HR_DIR}/{fname}', cv2.cvtColor(hr_np, cv2.COLOR_RGB2BGR))
            lr_np = cv2.resize(hr_np, (32, 32), interpolation=cv2.INTER_AREA)
            cv2.imwrite(f'{LR_DIR}/{fname}', cv2.cvtColor(lr_np, cv2.COLOR_RGB2BGR))
            crop_count += 1
        except Exception:
            continue

print(f'\\n\u2705 SR dataset: {crop_count} paired crops (HR 128x128, LR 32x32)')""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 3B \u2014 Train jersey_upscaler_v5 (SRResNet)
# ═══════════════════════════════════════════════════════
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Define exact same architecture as app/services/jersey_upscaler.py
class SRBlock(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), nn.BatchNorm2d(c), nn.PReLU(),
            nn.Conv2d(c, c, 3, 1, 1), nn.BatchNorm2d(c),
        )
    def forward(self, x):
        return x + self.block(x)

class JerseySR(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Sequential(nn.Conv2d(3, 64, 9, 1, 4), nn.PReLU())
        self.res = nn.Sequential(*[SRBlock(64) for _ in range(8)])
        self.mid = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))
        self.up = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(2), nn.PReLU(),
            nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(2), nn.PReLU(),
        )
        self.out = nn.Conv2d(64, 3, 9, 1, 4)

    def forward(self, x):
        e = self.entry(x)
        r = self.mid(self.res(e))
        return torch.clamp(self.out(self.up(e + r)), 0, 1)

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_files = sorted(glob.glob(f'{lr_dir}/*.png'))
        self.hr_dir = hr_dir
    def __len__(self):
        return len(self.lr_files)
    def __getitem__(self, idx):
        lr_path = self.lr_files[idx]
        hr_path = os.path.join(self.hr_dir, os.path.basename(lr_path))
        lr = cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        hr = cv2.cvtColor(cv2.imread(hr_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(lr).permute(2,0,1), torch.from_numpy(hr).permute(2,0,1)

SR_DIR = f'{DATA_BASE}/sr_dataset'
dataset = SRDataset(f'{SR_DIR}/lr', f'{SR_DIR}/hr')
assert len(dataset) >= 100, f'\u274c Need 100+ crops, have {len(dataset)}'

val_size = max(1, len(dataset) // 10)
train_size = len(dataset) - val_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
print(f'\u2705 SR Dataset: {train_size} train, {val_size} val')

model = JerseySR().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.L1Loss()

EPOCHS = 100; PATIENCE = 20
best_val_loss = float('inf')
patience_counter = 0

print(f'\\n{"=" * 60}')
print(f'\U0001f680 TRAINING: jersey_upscaler_v5 (SRResNet)')
print(f'   epochs={EPOCHS}, L1 loss, Adam lr=1e-4, patience={PATIENCE}')
print(f'{"=" * 60}\\n')

ckpt_path = f'{DRIVE_CHECKPOINTS}/jersey_upscaler_v5/checkpoint.pt'
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
start_epoch = 0
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    start_epoch = ckpt.get('epoch', 0) + 1
    best_val_loss = ckpt.get('best_val_loss', float('inf'))
    print(f'\u267b\ufe0f Resuming from epoch {start_epoch}')

for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = 0
    for lr_batch, hr_batch in train_dl:
        lr_batch, hr_batch = lr_batch.cuda(), hr_batch.cuda()
        pred = model(lr_batch)
        loss = criterion(pred, hr_batch)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dl)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for lr_batch, hr_batch in val_dl:
            lr_batch, hr_batch = lr_batch.cuda(), hr_batch.cuda()
            val_loss += criterion(model(lr_batch), hr_batch).item()
    val_loss /= len(val_dl)
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        print(f'  Epoch {epoch+1}/{EPOCHS} \u2014 train: {train_loss:.4f}, val: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss; patience_counter = 0
        torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                     'epoch': epoch, 'best_val_loss': best_val_loss},
                    ckpt_path.replace('checkpoint.pt', 'best.pt'))
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                     'epoch': epoch, 'best_val_loss': best_val_loss}, ckpt_path)

    if patience_counter >= PATIENCE:
        print(f'  Early stopping at epoch {epoch+1}'); break

print(f'\\n\u2705 jersey_upscaler_v5 training complete! Best val_loss: {best_val_loss:.4f}')""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 3C \u2014 Save jersey_upscaler_v5.pth to Drive
# ═══════════════════════════════════════════════════════
MODEL_NAME = 'jersey_upscaler_v5'
best_path = f'{DRIVE_CHECKPOINTS}/{MODEL_NAME}/best.pt'
assert os.path.exists(best_path), f'\u274c No best checkpoint \u2014 run Cell 3B first'
ckpt = torch.load(best_path)
dst = f'{DRIVE_SAVE_DIR}/{MODEL_NAME}.pth'
torch.save({'model_state': ckpt['model_state']}, dst)
sz = os.path.getsize(dst) / 1024 / 1024
print(f'{"=" * 60}')
print(f'\u2705 {MODEL_NAME}.pth SAVED TO DRIVE')
print(f'Location: {DRIVE_SAVE_DIR}/')
print(f'File size: {sz:.1f}MB')
print(f'Best val_loss: {ckpt.get("best_val_loss", "N/A")}')
print(f'{"=" * 60}')
print(f'\\nNote: This is a .pth file (PyTorch state_dict), NOT a YOLO .pt')""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 4: PLAYER ISOLATOR v3
# ═══════════════════════════════════════════════════════════════════
md("""\
---
## Section 4: Player Isolator v3
YOLO detect model \u2014 isolates individual players with tight bounding boxes, optimized as preprocessing for jersey OCR.
Output: `player_isolator_v3.pt`""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 4A \u2014 Download player isolation datasets
# ═══════════════════════════════════════════════════════
from roboflow import Roboflow

assert ROBOFLOW_API_KEY, '\u274c Set ROBOFLOW_API_KEY in Colab Secrets'

print('=== Downloading Player Isolation Datasets ===')
ds1 = safe_download('augmented-startups', 'football-player-detection-kucab', 1, 'football_iso')
ds2 = safe_download('roboflow-universe-projects', 'basketball-players-fy4c2', 1, 'basketball_iso')
ds3 = safe_download('ryseai', 'lacrosse-object-detection', 1, 'lacrosse_iso')
ds4 = safe_download('sports-person-detection', 'person-detection-sports', 1, 'sports_person')

print('\\nMerging player isolation datasets...')
MP = f'{DATA_BASE}/player_isolator_merged'
for s in ['train', 'val']:
    os.makedirs(f'{MP}/{s}/images', exist_ok=True)
    os.makedirs(f'{MP}/{s}/labels', exist_ok=True)

for ds in [ds1, ds2, ds3, ds4]:
    if not ds: continue
    loc = ds.location; dn = os.path.basename(loc)
    for sp in ['train', 'valid', 'val', 'test']:
        idir = f'{loc}/{sp}/images'; ldir = f'{loc}/{sp}/labels'
        if not os.path.exists(idir): continue
        target = 'val' if sp in ['valid', 'val', 'test'] else 'train'
        for img_path in glob.glob(f'{idir}/*'):
            fn = f'{dn}_{os.path.basename(img_path)}'
            shutil.copy2(img_path, f'{MP}/{target}/images/{fn}')
            lbl = os.path.join(ldir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
            if os.path.exists(lbl):
                with open(lbl) as f:
                    lines = ['0 ' + ' '.join(l.strip().split()[1:]) for l in f if len(l.strip().split()) >= 5]
                if lines:
                    with open(f'{MP}/{target}/labels/{os.path.splitext(fn)[0]}.txt', 'w') as f:
                        f.write('\\n'.join(lines) + '\\n')

dy = {'path': MP, 'train': 'train/images', 'val': 'val/images', 'nc': 1, 'names': {0: 'player'}}
with open(f'{MP}/data.yaml', 'w') as f: yaml.dump(dy, f)
tc = len(glob.glob(f'{MP}/train/images/*')); vc = len(glob.glob(f'{MP}/val/images/*'))
print(f'\\n\u2705 Player isolator dataset: {tc} train, {vc} val')""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 4B \u2014 Train player_isolator_v3
# ═══════════════════════════════════════════════════════
from ultralytics import YOLO

MODEL_NAME = 'player_isolator_v3'
DATA_PATH = f'{DATA_BASE}/player_isolator_merged/data.yaml'
assert os.path.exists(DATA_PATH), '\u274c No data \u2014 run Cell 4A first'
tc = len(glob.glob(f'{DATA_BASE}/player_isolator_merged/train/images/*'))
assert tc >= 50, f'\u274c Need 50+ images, have {tc}'
print(f'\u2705 {tc} training images ready')

print(f'\\n{"=" * 60}')
print(f'\U0001f680 TRAINING: {MODEL_NAME}')
print(f'   epochs=100, imgsz=640, optimized for tight player bboxes')
print(f'{"=" * 60}\\n')

resume_path = f'{DRIVE_CHECKPOINTS}/{MODEL_NAME}/last.pt'
if os.path.exists(resume_path) and os.path.getsize(resume_path) > 1024*1024:
    print('\u267b\ufe0f Resuming...'); model = YOLO(resume_path); model.train(resume=True)
else:
    model = YOLO('yolov8m.pt')
    model.train(data=DATA_PATH, epochs=100, imgsz=640, batch=DEFAULT_BATCH,
                name=MODEL_NAME, device=0, patience=20, save_period=5,
                amp=True, cache=True, mosaic=0.8, scale=0.4)

backup_checkpoints(MODEL_NAME, 'detect')
print(f'\\n\u2705 {MODEL_NAME} training complete!')""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 4C \u2014 Save player_isolator_v3.pt to Drive
# ═══════════════════════════════════════════════════════
save_model_to_drive('player_isolator_v3', 'detect')""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 5: BASKETBALL JERSEY OCR v3
# ═══════════════════════════════════════════════════════════════════
md("""\
---
## Section 5: Basketball Jersey Number OCR v3
YOLO detect model \u2014 basketball-specific jersey number detector (digits 0\u20139).
Output: `basketball_jersey_number_v3.pt`""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 5A \u2014 Download basketball jersey OCR datasets
# ═══════════════════════════════════════════════════════
from roboflow import Roboflow

assert ROBOFLOW_API_KEY, '\u274c Set ROBOFLOW_API_KEY in Colab Secrets'

print('=== Downloading Basketball Jersey OCR Datasets ===')
ds1 = safe_download('roboflow-universe-projects', 'jersey-number-detection-2', 1, 'jersey_ocr')
ds2 = safe_download('basketball-jersey-number', 'basketball-jersey', 1, 'bball_jersey')
ds3 = safe_download('augmented-startups', 'basketball-players-fy4c2', 1, 'bball_players')

print('\\nMerging basketball OCR datasets...')
MP = f'{DATA_BASE}/basketball_ocr_v3_merged'
for s in ['train', 'val']:
    os.makedirs(f'{MP}/{s}/images', exist_ok=True)
    os.makedirs(f'{MP}/{s}/labels', exist_ok=True)

NC = 10  # digits 0-9
for ds in [ds1, ds2, ds3]:
    if not ds: continue
    loc = ds.location; dn = os.path.basename(loc)
    for sp in ['train', 'valid', 'val', 'test']:
        idir = f'{loc}/{sp}/images'; ldir = f'{loc}/{sp}/labels'
        if not os.path.exists(idir): continue
        target = 'val' if sp in ['valid', 'val', 'test'] else 'train'
        for img_path in glob.glob(f'{idir}/*'):
            fn = f'{dn}_{os.path.basename(img_path)}'
            shutil.copy2(img_path, f'{MP}/{target}/images/{fn}')
            lbl = os.path.join(ldir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
            if os.path.exists(lbl):
                with open(lbl) as f:
                    lines = []
                    for l in f:
                        parts = l.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            mapped = cls_id if cls_id < 10 else 0
                            lines.append(f'{mapped} ' + ' '.join(parts[1:]))
                if lines:
                    with open(f'{MP}/{target}/labels/{os.path.splitext(fn)[0]}.txt', 'w') as f:
                        f.write('\\n'.join(lines) + '\\n')

dy = {'path': MP, 'train': 'train/images', 'val': 'val/images', 'nc': NC, 'names': {i: str(i) for i in range(NC)}}
with open(f'{MP}/data.yaml', 'w') as f: yaml.dump(dy, f)
tc = len(glob.glob(f'{MP}/train/images/*')); vc = len(glob.glob(f'{MP}/val/images/*'))
print(f'\\n\u2705 Basketball OCR v3 dataset: {tc} train, {vc} val \u2014 {NC} classes')""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 5B \u2014 Train basketball_jersey_number_v3
# ═══════════════════════════════════════════════════════
from ultralytics import YOLO

MODEL_NAME = 'basketball_jersey_number_v3'
DATA_PATH = f'{DATA_BASE}/basketball_ocr_v3_merged/data.yaml'
assert os.path.exists(DATA_PATH), '\u274c No data \u2014 run Cell 5A first'
tc = len(glob.glob(f'{DATA_BASE}/basketball_ocr_v3_merged/train/images/*'))
assert tc >= 50, f'\u274c Need 50+ images, have {tc}'
print(f'\u2705 {tc} training images ready')

print(f'\\n{"=" * 60}')
print(f'\U0001f680 TRAINING: {MODEL_NAME}')
print(f'   epochs=120, fliplr=0 (digits), patience=25')
print(f'{"=" * 60}\\n')

resume_path = f'{DRIVE_CHECKPOINTS}/{MODEL_NAME}/last.pt'
if os.path.exists(resume_path) and os.path.getsize(resume_path) > 1024*1024:
    print('\u267b\ufe0f Resuming...'); model = YOLO(resume_path); model.train(resume=True)
else:
    model = YOLO('yolov8m.pt')
    model.train(data=DATA_PATH, epochs=120, imgsz=640, batch=DEFAULT_BATCH,
                name=MODEL_NAME, device=0, patience=25, save_period=5,
                amp=True, cache=True, fliplr=0.0, mosaic=0.5, degrees=5.0, scale=0.3)

backup_checkpoints(MODEL_NAME, 'detect')
print(f'\\n\u2705 {MODEL_NAME} training complete!')""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 5C \u2014 Save basketball_jersey_number_v3.pt to Drive
# ═══════════════════════════════════════════════════════
save_model_to_drive('basketball_jersey_number_v3', 'detect')""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 6: BASKETBALL PLAYER DETECTOR v2
# ═══════════════════════════════════════════════════════════════════
md("""\
---
## Section 6: Basketball Player Detector v2
YOLO detect model \u2014 basketball-specific player detection, optimized for indoor courts.
Output: `basketball_player_detector_v2.pt`""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 6A \u2014 Download basketball player detection datasets
# ═══════════════════════════════════════════════════════
from roboflow import Roboflow

assert ROBOFLOW_API_KEY, '\u274c Set ROBOFLOW_API_KEY in Colab Secrets'

print('=== Downloading Basketball Player Detection Datasets ===')
ds1 = safe_download('roboflow-universe-projects', 'basketball-players-fy4c2', 1, 'bball_1')
ds2 = safe_download('basketball-detect', 'basketball-player-detection', 1, 'bball_2')
ds3 = safe_download('basketball-ai', 'basketball-player-detector', 1, 'bball_3')

print('\\nMerging basketball player datasets...')
MP = f'{DATA_BASE}/basketball_player_v2_merged'
for s in ['train', 'val']:
    os.makedirs(f'{MP}/{s}/images', exist_ok=True)
    os.makedirs(f'{MP}/{s}/labels', exist_ok=True)

for ds in [ds1, ds2, ds3]:
    if not ds: continue
    loc = ds.location; dn = os.path.basename(loc)
    for sp in ['train', 'valid', 'val', 'test']:
        idir = f'{loc}/{sp}/images'; ldir = f'{loc}/{sp}/labels'
        if not os.path.exists(idir): continue
        target = 'val' if sp in ['valid', 'val', 'test'] else 'train'
        for img_path in glob.glob(f'{idir}/*'):
            fn = f'{dn}_{os.path.basename(img_path)}'
            shutil.copy2(img_path, f'{MP}/{target}/images/{fn}')
            lbl = os.path.join(ldir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
            if os.path.exists(lbl):
                with open(lbl) as f:
                    lines = ['0 ' + ' '.join(l.strip().split()[1:]) for l in f if len(l.strip().split()) >= 5]
                if lines:
                    with open(f'{MP}/{target}/labels/{os.path.splitext(fn)[0]}.txt', 'w') as f:
                        f.write('\\n'.join(lines) + '\\n')

dy = {'path': MP, 'train': 'train/images', 'val': 'val/images', 'nc': 1, 'names': {0: 'basketball_player'}}
with open(f'{MP}/data.yaml', 'w') as f: yaml.dump(dy, f)
tc = len(glob.glob(f'{MP}/train/images/*')); vc = len(glob.glob(f'{MP}/val/images/*'))
print(f'\\n\u2705 Basketball player v2 dataset: {tc} train, {vc} val')""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 6B \u2014 Train basketball_player_detector_v2
# ═══════════════════════════════════════════════════════
from ultralytics import YOLO

MODEL_NAME = 'basketball_player_detector_v2'
DATA_PATH = f'{DATA_BASE}/basketball_player_v2_merged/data.yaml'
assert os.path.exists(DATA_PATH), '\u274c No data \u2014 run Cell 6A first'
tc = len(glob.glob(f'{DATA_BASE}/basketball_player_v2_merged/train/images/*'))
assert tc >= 30, f'\u274c Need 30+ images, have {tc}'
print(f'\u2705 {tc} training images ready')

print(f'\\n{"=" * 60}')
print(f'\U0001f680 TRAINING: {MODEL_NAME}')
print(f'   epochs=100, yolov8n (nano for speed)')
print(f'{"=" * 60}\\n')

resume_path = f'{DRIVE_CHECKPOINTS}/{MODEL_NAME}/last.pt'
if os.path.exists(resume_path) and os.path.getsize(resume_path) > 1024*1024:
    print('\u267b\ufe0f Resuming...'); model = YOLO(resume_path); model.train(resume=True)
else:
    model = YOLO('yolov8n.pt')
    model.train(data=DATA_PATH, epochs=100, imgsz=640, batch=DEFAULT_BATCH,
                name=MODEL_NAME, device=0, patience=20, save_period=5,
                amp=True, cache=True)

backup_checkpoints(MODEL_NAME, 'detect')
print(f'\\n\u2705 {MODEL_NAME} training complete!')""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 6C \u2014 Save basketball_player_detector_v2.pt to Drive
# ═══════════════════════════════════════════════════════
save_model_to_drive('basketball_player_detector_v2', 'detect')""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 7: LACROSSE DETECTOR v2
# ═══════════════════════════════════════════════════════════════════
md("""\
---
## Section 7: Lacrosse Detector v2
YOLO detect model \u2014 lacrosse-specific player detection.
Output: `lacrosse_detector_v2.pt`""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 7A \u2014 Download lacrosse detection datasets
# ═══════════════════════════════════════════════════════
from roboflow import Roboflow

assert ROBOFLOW_API_KEY, '\u274c Set ROBOFLOW_API_KEY in Colab Secrets'

print('=== Downloading Lacrosse Detection Datasets ===')
ds1 = safe_download('ryseai', 'lacrosse-object-detection', 1, 'lax_1')
ds2 = safe_download('lacrosse-detect', 'lacrosse-player-detection', 1, 'lax_2')
ds3 = safe_download('lacrosse-ai', 'lacrosse-game-detection', 1, 'lax_3')

print('\\nMerging lacrosse datasets...')
MP = f'{DATA_BASE}/lacrosse_v2_merged'
for s in ['train', 'val']:
    os.makedirs(f'{MP}/{s}/images', exist_ok=True)
    os.makedirs(f'{MP}/{s}/labels', exist_ok=True)

for ds in [ds1, ds2, ds3]:
    if not ds: continue
    loc = ds.location; dn = os.path.basename(loc)
    for sp in ['train', 'valid', 'val', 'test']:
        idir = f'{loc}/{sp}/images'; ldir = f'{loc}/{sp}/labels'
        if not os.path.exists(idir): continue
        target = 'val' if sp in ['valid', 'val', 'test'] else 'train'
        for img_path in glob.glob(f'{idir}/*'):
            fn = f'{dn}_{os.path.basename(img_path)}'
            shutil.copy2(img_path, f'{MP}/{target}/images/{fn}')
            lbl = os.path.join(ldir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
            if os.path.exists(lbl):
                with open(lbl) as f:
                    lines = ['0 ' + ' '.join(l.strip().split()[1:]) for l in f if len(l.strip().split()) >= 5]
                if lines:
                    with open(f'{MP}/{target}/labels/{os.path.splitext(fn)[0]}.txt', 'w') as f:
                        f.write('\\n'.join(lines) + '\\n')

dy = {'path': MP, 'train': 'train/images', 'val': 'val/images', 'nc': 1, 'names': {0: 'lacrosse_player'}}
with open(f'{MP}/data.yaml', 'w') as f: yaml.dump(dy, f)
tc = len(glob.glob(f'{MP}/train/images/*')); vc = len(glob.glob(f'{MP}/val/images/*'))
print(f'\\n\u2705 Lacrosse v2 dataset: {tc} train, {vc} val')""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 7B \u2014 Train lacrosse_detector_v2
# ═══════════════════════════════════════════════════════
from ultralytics import YOLO

MODEL_NAME = 'lacrosse_detector_v2'
DATA_PATH = f'{DATA_BASE}/lacrosse_v2_merged/data.yaml'
assert os.path.exists(DATA_PATH), '\u274c No data \u2014 run Cell 7A first'
tc = len(glob.glob(f'{DATA_BASE}/lacrosse_v2_merged/train/images/*'))
assert tc >= 30, f'\u274c Need 30+ images, have {tc}'
print(f'\u2705 {tc} training images ready')

print(f'\\n{"=" * 60}')
print(f'\U0001f680 TRAINING: {MODEL_NAME}')
print(f'   epochs=100, yolov8n (nano)')
print(f'{"=" * 60}\\n')

resume_path = f'{DRIVE_CHECKPOINTS}/{MODEL_NAME}/last.pt'
if os.path.exists(resume_path) and os.path.getsize(resume_path) > 1024*1024:
    print('\u267b\ufe0f Resuming...'); model = YOLO(resume_path); model.train(resume=True)
else:
    model = YOLO('yolov8n.pt')
    model.train(data=DATA_PATH, epochs=100, imgsz=640, batch=DEFAULT_BATCH,
                name=MODEL_NAME, device=0, patience=20, save_period=5,
                amp=True, cache=True)

backup_checkpoints(MODEL_NAME, 'detect')
print(f'\\n\u2705 {MODEL_NAME} training complete!')""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 7C \u2014 Save lacrosse_detector_v2.pt to Drive
# ═══════════════════════════════════════════════════════
save_model_to_drive('lacrosse_detector_v2', 'detect')""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 8: FINAL REPORT
# ═══════════════════════════════════════════════════════════════════
md("""\
---
## Section 8: Final Report
Verify all 7 models are saved to Google Drive and print copy instructions.""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 8A \u2014 Check Drive folder
# ═══════════════════════════════════════════════════════
print(f'\\nChecking {DRIVE_SAVE_DIR}...\\n')
all_files = sorted(glob.glob(f'{DRIVE_SAVE_DIR}/*.*'))
if all_files:
    total_size = 0
    for f in all_files:
        sz = os.path.getsize(f) / 1024 / 1024; total_size += sz
        print(f'  {os.path.basename(f):45s} {sz:8.1f}MB')
    print(f'\\n  Total: {len(all_files)} files, {total_size:.1f}MB')
else:
    print('  No files found \u2014 run Sections 1-7')""")

code("""\
# ═══════════════════════════════════════════════════════
# Cell 8B \u2014 Final Status Table
# ═══════════════════════════════════════════════════════
ALL_MODELS = [
    ('scoreboard_detector_v5.pt', 'YOLO detect', 'Scoreboard detection'),
    ('dead_ball_classifier_v5.pt', 'YOLO classify', 'Dead ball vs live play'),
    ('jersey_upscaler_v5.pth', 'PyTorch SR', 'Jersey crop 4x upscale'),
    ('player_isolator_v3.pt', 'YOLO detect', 'Player isolation for OCR'),
    ('basketball_jersey_number_v3.pt', 'YOLO detect', 'Basketball jersey OCR'),
    ('basketball_player_detector_v2.pt', 'YOLO detect', 'Basketball player detection'),
    ('lacrosse_detector_v2.pt', 'YOLO detect', 'Lacrosse player detection'),
]

print('=' * 70)
print('  CLIPT V6 TRAINING COMPLETE \u2014 THE LAST NOTEBOOK')
print('=' * 70)

print(f'\\n\U0001f4be GOOGLE DRIVE ({DRIVE_SAVE_DIR}):')
print(f'   \u2192 Open Google Drive \u2192 MyDrive \u2192 clipt_v6_models')
print(f'   \u2192 Copy ALL files to: reelapp/playerJerseyIdentification-master/app/model/')

saved = 0; missing = 0
print(f'\\n{"Model":<40s} {"Type":<15s} {"Status":<10s} {"Size":<10s}')
print('\u2500' * 75)

for mn, mtype, desc in ALL_MODELS:
    dp = f'{DRIVE_SAVE_DIR}/{mn}'
    if os.path.exists(dp):
        sz = os.path.getsize(dp) / 1024 / 1024
        print(f'{mn:<40s} {mtype:<15s} \u2705 SAVED    {sz:>7.1f}MB')
        saved += 1
    else:
        print(f'{mn:<40s} {mtype:<15s} \u274c MISSING')
        missing += 1

print('\u2500' * 75)
print(f'\\n\u2705 {saved}/{len(ALL_MODELS)} models saved')
if missing > 0:
    print(f'\u274c {missing} models still need training')
else:
    print(f'\\n\U0001f389 ALL MODELS TRAINED!')

print(f'\\n\U0001f4cb NEXT STEPS:')
print(f'   1. Download all files from Google Drive \u2192 clipt_v6_models/')
print(f'   2. Copy to: reelapp/playerJerseyIdentification-master/app/model/')
print(f'   3. Run the MEGA_PROMPT_V6.md in Claude Code')
print(f'   4. Deploy to Railway')
print(f'{"=" * 70}')""")

# ═══════════════════════════════════════════════════════════════════
# WRITE NOTEBOOK
# ═══════════════════════════════════════════════════════════════════
nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
        "colab": {"provenance": [], "gpuType": "A100"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out = "train_models_v6_final.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Wrote {out} with {len(cells)} cells")
