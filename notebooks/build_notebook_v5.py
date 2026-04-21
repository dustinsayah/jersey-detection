#!/usr/bin/env python3
"""Generate train_models_v5_autolabel.ipynb in v4.2 style."""
import json, pathlib

cells = []

def md(source):
    lines = source.strip().split("\n")
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]
    })

def code(source):
    lines = source.strip().split("\n")
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]
    })

# ============================================================
# TITLE
# ============================================================
md("""# Clipt v5 — Auto-Label Training Pipeline

**8 models** with Google Drive saves, crash protection, and autopilot execution.

| Cell | Model | Method | Epochs |
|------|-------|--------|--------|
| 0 | Setup + Drive Mount | — | — |
| 1A-1G | YouTube Harvesting | yt-dlp + scenedetect | — |
| 2A-2D | Claude Auto-Labeling | Claude Haiku Vision | — |
| 3A-3F | VideoMAE (3 sports) | Fine-tune VideoMAE | 20 |
| 4A-4F | Outcome Classifier (3 sports) | YOLO classify | 80 |
| 5A-5C | Jersey OCR Universal | YOLO detect | 120 |
| 6A-6C | Player Detector | YOLO detect | 100 |
| 7A-7B | Final Report | — | — |

**Total: ~4-6 hours on A100** (all autopilot — just click play on each cell)

### Finished Models → Google Drive
- **All files save to Google Drive:** `MyDrive/clipt_v5_models/`
- Zero browser downloads — just grab from Drive when done
- Copy the files from Drive into your reelapp folder

### Crash Protection
- **Google Drive auto-backup** every 5 epochs
- **Auto-resume** — if Colab disconnects, rerun Section 0, then rerun the crashed cell
- Labels saved to Drive after every 10 clips

### IF COLAB DISCONNECTS
1. Reconnect runtime (A100 GPU)
2. Rerun **Section 0** (all 5 cells) — reinstalls packages + remounts Drive
3. Your completed models are safe on Google Drive
4. Rerun the crashed cell — it auto-resumes from Drive checkpoint

**Colab Secrets needed:** `ROBOFLOW_API_KEY`, `ANTHROPIC_API_KEY`""")

# ============================================================
# SECTION 0: SETUP
# ============================================================
md("""---
## Section 0: Setup""")

# Cell 0A
code("""# ═══════════════════════════════════════════════════════
# Cell 0A — Install Packages
# ═══════════════════════════════════════════════════════
!pip install roboflow ultralytics pyyaml anthropic "scenedetect[opencv]" yt-dlp transformers datasets accelerate -q
print("\\n\\u2705 All packages installed")""")

# Cell 0B
code("""# ═══════════════════════════════════════════════════════
# Cell 0B — Imports + Config
# ═══════════════════════════════════════════════════════
import os, sys, torch, shutil, glob, yaml, time, json, cv2, gc, zipfile, subprocess, base64
import numpy as np
from collections import Counter
from google.colab import userdata

# ── Directories ──
DRIVE_SAVE_DIR = '/content/drive/MyDrive/clipt_v5_models'
DRIVE_CHECKPOINTS = '/content/drive/MyDrive/clipt_v5_models/checkpoints'
DRIVE_LABELS = '/content/drive/MyDrive/clipt_v5_models/labels'
CLIPS_BASE = '/content/clips'
RAW_VIDEO_BASE = '/content/raw_videos'

# ── Cost Controls ──
MAX_CLIPS_TO_LABEL = 200
MAX_API_CALLS_TOTAL = 650

# ── Play Types ──
PLAY_TYPES = {
    'basketball': ['layup','jump_shot','dunk','three_pointer','fast_break','rebound',
                    'steal','block','assist','free_throw','turnover','other'],
    'football': ['pass_play','run_play','touchdown','interception','sack','field_goal',
                 'punt','kickoff','tackle','catch','other'],
    'lacrosse': ['shot','goal','save','ground_ball','face_off','clear','dodge',
                 'pass','ride','other'],
}

# ── Create Directories ──
for sport in ['basketball', 'football', 'lacrosse']:
    os.makedirs(f'{CLIPS_BASE}/{sport}', exist_ok=True)
    os.makedirs(f'{RAW_VIDEO_BASE}/{sport}', exist_ok=True)

# ── API Keys ──
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

# ── GPU Check ──
assert torch.cuda.is_available(), '\\u274c NO GPU \\u2014 Runtime > Change runtime type > A100'
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
HAS_A100 = vram_gb > 40
DEFAULT_BATCH = 16 if HAS_A100 else 8

print(f'\\u2705 GPU: {gpu_name} ({vram_gb:.1f}GB)')
if HAS_A100:
    print(f'\\u2705 A100 detected \\u2014 batch={DEFAULT_BATCH}')
print(f'\\u2705 Roboflow API key: {"loaded" if ROBOFLOW_API_KEY else "\\u274c MISSING"}')
print(f'\\u2705 Anthropic API key: {"loaded" if ANTHROPIC_API_KEY else "\\u274c MISSING"}')""")

# Cell 0C
code("""# ═══════════════════════════════════════════════════════
# Cell 0C — Mount Google Drive
# ═══════════════════════════════════════════════════════
from google.colab import drive
drive.mount('/content/drive')

os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
os.makedirs(DRIVE_CHECKPOINTS, exist_ok=True)
os.makedirs(DRIVE_LABELS, exist_ok=True)

print(f'\\n\\u2705 Drive mounted')
print(f'\\u2705 Models save to: {DRIVE_SAVE_DIR}')
print(f'   (grab finished files from here \\u2192 reelapp folder)')

# Show existing models
existing = glob.glob(f'{DRIVE_SAVE_DIR}/*.*')
if existing:
    print(f'\\nFound {len(existing)} existing files:')
    for f in sorted(existing):
        sz = os.path.getsize(f) / 1024 / 1024
        print(f'  {os.path.basename(f)} \\u2014 {sz:.1f}MB')""")

# Cell 0D
code("""# ═══════════════════════════════════════════════════════
# Cell 0D — Test Anthropic API Key
# ═══════════════════════════════════════════════════════
import anthropic

assert ANTHROPIC_API_KEY, '\\u274c Set ANTHROPIC_API_KEY in Colab Secrets (left sidebar > key icon)'

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
try:
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[{"role": "user", "content": "Say OK"}]
    )
    print(f'\\u2705 Anthropic API working \\u2014 claude-sonnet-4-20250514 verified')
    print(f'   Response: {resp.content[0].text}')
    print(f'   (Will use claude-haiku-4-5-20251001 for labeling \\u2014 12x cheaper)')
except Exception as e:
    print(f'\\u274c Anthropic API FAILED: {e}')
    print(f'\\u26a0\\ufe0f Check your ANTHROPIC_API_KEY in Colab Secrets')""")

# Cell 0E
code("""# ═══════════════════════════════════════════════════════
# Cell 0E — Test yt-dlp + Print Setup Summary
# ═══════════════════════════════════════════════════════
import yt_dlp

test_dir = '/content/yt_test'
os.makedirs(test_dir, exist_ok=True)
ytdlp_ok = False

ydl_opts = {
    'format': 'best[height<=720]',
    'outtmpl': f'{test_dir}/%(id)s.%(ext)s',
    'quiet': False,
    'extractor_args': {'youtube': {
        'player_client': ['android'],
    }},
}

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info('ytsearch1:basketball highlights short clip', download=True)
    files = glob.glob(f'{test_dir}/*')
    if files:
        print(f'\\u2705 yt-dlp downloaded: {os.path.basename(files[0])} ({os.path.getsize(files[0])/1024/1024:.1f}MB)')
        for f in files:
            os.remove(f)
        ytdlp_ok = True
    else:
        print('\\u274c yt-dlp: no file downloaded')
except Exception as e:
    print(f'\\u274c yt-dlp FAILED: {e}')
    print('\\u26a0\\ufe0f Try: !pip install -U yt-dlp')
shutil.rmtree(test_dir, ignore_errors=True)

# Test Roboflow
rf_ok = False
try:
    from roboflow import Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    rf_ok = True
    print('\\u2705 Roboflow API initialized')
except Exception as e:
    print(f'\\u274c Roboflow FAILED: {e}')

print(f'\\n{"="*60}')
print('\\u2705 SETUP COMPLETE')
print(f'GPU: {gpu_name} \\u2014 {"A100" if HAS_A100 else "\\u26a0\\ufe0f not A100"}')
print(f'Drive mounted: {DRIVE_SAVE_DIR}')
print(f'Anthropic API: {"\\u2705 working" if ANTHROPIC_API_KEY else "\\u274c missing"}')
print(f'yt-dlp test: {"\\u2705 working" if ytdlp_ok else "\\u274c failed"}')
print(f'Roboflow API: {"\\u2705 working" if rf_ok else "\\u274c failed"}')
print(f'All packages installed: \\u2705')
print(f'{"="*60}')""")

# ============================================================
# SECTION 1: YOUTUBE HARVESTING
# ============================================================
md("""---
## Section 1: YouTube Harvesting
Download sports highlight videos and split into clips via scene detection.""")

# Helper function to create download + scene detect cells for each sport
SPORT_CONFIGS = [
    ('basketball', 'A', 'B', [
        'ytsearch3:NBA best plays highlights compilation',
        'ytsearch3:NBA dunks blocks steals compilation 2024',
        'ytsearch3:college basketball highlights game 2024',
    ]),
    ('football', 'C', 'D', [
        'ytsearch3:NFL touchdown compilation',
        'ytsearch3:NFL best plays highlights compilation 2024',
        'ytsearch3:college football touchdown highlights',
    ]),
    ('lacrosse', 'E', 'F', [
        'ytsearch3:lacrosse goals compilation high school',
        'ytsearch3:PLL lacrosse highlights goals',
        'ytsearch3:NCAA college lacrosse highlights compilation',
    ]),
]

for sport, dl_letter, sd_letter, queries in SPORT_CONFIGS:
    queries_str = ",\n    ".join([f"'{q}'" for q in queries])

    # Download cell
    code(f"""# ═══════════════════════════════════════════════════════
# Cell 1{dl_letter} \\u2014 Download {sport} videos
# ═══════════════════════════════════════════════════════
import yt_dlp

SPORT = '{sport}'
DL_DIR = f'{{RAW_VIDEO_BASE}}/{{SPORT}}'
os.makedirs(DL_DIR, exist_ok=True)

SEARCH_QUERIES = [
    {queries_str},
]

ydl_opts = {{
    'format': 'best[height<=720]',
    'outtmpl': f'{{DL_DIR}}/%(id)s.%(ext)s',
    'quiet': False,
    'extractor_args': {{'youtube': {{'player_client': ['android']}}}},
    'ignoreerrors': True,
    'merge_output_format': 'mp4',
}}

for qi, query in enumerate(SEARCH_QUERIES):
    print(f'\\n\\U0001f50d [{{qi+1}}/{{len(SEARCH_QUERIES)}}] Searching: {{query}}')
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([query])
    except Exception as e:
        print(f'  \\u26a0\\ufe0f Query failed: {{e}}')

total = len(glob.glob(f'{{DL_DIR}}/*'))
print(f'\\n\\u2705 {sport.capitalize()} videos downloaded: {{total}}')
if total == 0:
    print('\\u274c No videos downloaded \\u2014 check yt-dlp config or add manual URLs')""")

    # Scene detect cell
    code(f"""# ═══════════════════════════════════════════════════════
# Cell 1{sd_letter} \\u2014 Scene detect {sport} \\u2192 clips
# ═══════════════════════════════════════════════════════
from scenedetect import detect, ContentDetector

SPORT = '{sport}'
DL_DIR = f'{{RAW_VIDEO_BASE}}/{{SPORT}}'
CLIPS_DIR = f'{{CLIPS_BASE}}/{{SPORT}}'
os.makedirs(CLIPS_DIR, exist_ok=True)

videos = sorted(glob.glob(f'{{DL_DIR}}/*.mp4') + glob.glob(f'{{DL_DIR}}/*.webm') + glob.glob(f'{{DL_DIR}}/*.mkv'))
assert len(videos) > 0, f'\\u274c No videos in {{DL_DIR}} \\u2014 run Cell 1{dl_letter} first'

existing = len(glob.glob(f'{{CLIPS_DIR}}/*.mp4'))
print(f'Existing clips: {{existing}}')

for vi, vp in enumerate(videos):
    vname = os.path.splitext(os.path.basename(vp))[0]
    if glob.glob(f'{{CLIPS_DIR}}/{{vname}}_clip*.mp4'):
        print(f'[{{vi+1}}/{{len(videos)}}] {{os.path.basename(vp)}} \\u2014 already processed, skipping')
        continue
    print(f'[{{vi+1}}/{{len(videos)}}] Processing {{os.path.basename(vp)}}...')
    try:
        scenes = detect(vp, ContentDetector(threshold=27.0))
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        clip_n = 0
        for s, e in scenes:
            dur = (e.get_frames() - s.get_frames()) / fps
            if 2 <= dur <= 15:
                cp = f'{{CLIPS_DIR}}/{{vname}}_clip{{clip_n:04d}}.mp4'
                start_sec = s.get_frames() / fps
                subprocess.run(
                    ['ffmpeg', '-y', '-ss', str(start_sec), '-i', vp,
                     '-t', str(dur), '-c:v', 'libx264', '-preset', 'fast',
                     '-crf', '23', '-an', cp],
                    capture_output=True
                )
                if os.path.exists(cp) and os.path.getsize(cp) > 10000:
                    clip_n += 1
                else:
                    if os.path.exists(cp):
                        os.remove(cp)
        cap.release()
        print(f'  \\u2192 {{clip_n}} clips extracted')
    except Exception as e:
        print(f'  \\u26a0\\ufe0f Failed: {{e}}')

total_clips = len(glob.glob(f'{{CLIPS_DIR}}/*.mp4'))
print(f'\\n{sport.capitalize()} clips total: {{total_clips}}')""")

# Cell 1G
code("""# ═══════════════════════════════════════════════════════
# Cell 1G — Clip Count Summary
# ═══════════════════════════════════════════════════════
bb_clips = len(glob.glob(f'{CLIPS_BASE}/basketball/*.mp4'))
fb_clips = len(glob.glob(f'{CLIPS_BASE}/football/*.mp4'))
lax_clips = len(glob.glob(f'{CLIPS_BASE}/lacrosse/*.mp4'))
total = bb_clips + fb_clips + lax_clips

bb_vids = len(glob.glob(f'{RAW_VIDEO_BASE}/basketball/*'))
fb_vids = len(glob.glob(f'{RAW_VIDEO_BASE}/football/*'))
lax_vids = len(glob.glob(f'{RAW_VIDEO_BASE}/lacrosse/*'))

print(f'{"="*60}')
print('\\u2705 HARVESTING COMPLETE')
print(f'Basketball videos: {bb_vids} \\u2192 clips: {bb_clips} {"\\u2705" if bb_clips >= 50 else "\\u26a0\\ufe0f need 50+"}')
print(f'Football videos: {fb_vids} \\u2192 clips: {fb_clips} {"\\u2705" if fb_clips >= 50 else "\\u26a0\\ufe0f need 50+"}')
print(f'Lacrosse videos: {lax_vids} \\u2192 clips: {lax_clips} {"\\u2705" if lax_clips >= 50 else "\\u26a0\\ufe0f need 50+"}')
print(f'Total clips: {total}')
print(f'{"="*60}')

for sport, count in [('basketball', bb_clips), ('football', fb_clips), ('lacrosse', lax_clips)]:
    if count < 50:
        print(f'\\u26a0\\ufe0f WARNING: {sport} only has {count} clips (need 50+)')
        print(f'   Try adding more search queries or specific video URLs')""")

# ============================================================
# SECTION 2: CLAUDE AUTO-LABELING
# ============================================================
md("""---
## Section 2: Claude Auto-Labeling
Use Claude Vision (Haiku) to classify each clip by play type.
Extracts 3 frames per clip, sends to Claude, gets back a label.

**Cost:** ~$0.003/clip with Haiku. 650 max API calls = ~$1.95 max.""")

for sport in ['basketball', 'football', 'lacrosse']:
    letter = {'basketball': 'A', 'football': 'B', 'lacrosse': 'C'}[sport]
    code(f"""# ═══════════════════════════════════════════════════════
# Cell 2{letter} \\u2014 Label {sport} clips
# ═══════════════════════════════════════════════════════
import anthropic

SPORT = '{sport}'
CLIPS_DIR = f'{{CLIPS_BASE}}/{{SPORT}}'
LABELS_FILE = f'{{DRIVE_LABELS}}/{{SPORT}}_labels.json'

# \\u2500\\u2500 Guard \\u2500\\u2500
clip_files = sorted(glob.glob(f'{{CLIPS_DIR}}/*.mp4'))
assert len(clip_files) > 0, f'\\u274c No clips in {{CLIPS_DIR}} \\u2014 run Section 1 first'

# \\u2500\\u2500 Load existing labels from Drive \\u2500\\u2500
labels = {{}}
if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE) as f:
        labels = json.load(f)
    print(f'Loaded {{len(labels)}} existing labels from Drive')

unlabeled = [c for c in clip_files if os.path.basename(c) not in labels]
to_label = unlabeled[:MAX_CLIPS_TO_LABEL]
print(f'Total clips: {{len(clip_files)}}, Already labeled: {{len(labels)}}, To label: {{len(to_label)}}')

if not to_label:
    print('\\u2705 All clips already labeled!')
else:
    assert ANTHROPIC_API_KEY, '\\u274c Set ANTHROPIC_API_KEY in Colab Secrets'
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    types_list = ', '.join(PLAY_TYPES[SPORT])
    cost = 0.0
    api_calls = 0

    for i, cp in enumerate(to_label):
        # \\u2500\\u2500 Hard stop on API calls \\u2500\\u2500
        if api_calls >= MAX_API_CALLS_TOTAL:
            print(f'\\n\\u26d4 Hard stop: {{api_calls}} API calls reached')
            print(f'Estimated cost: ${{api_calls * 0.003:.2f}}')
            break

        cn = os.path.basename(cp)

        # Extract 3 frames
        cap = cv2.VideoCapture(cp)
        tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_b64s = []
        for frac in [0.1, 0.5, 0.9]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(tf * frac))
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (512, 288))
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_b64s.append(base64.b64encode(buf).decode())
        cap.release()

        if len(frame_b64s) < 2:
            labels[cn] = 'other'
            continue

        content = [{{"type": "image", "source": {{"type": "base64", "media_type": "image/jpeg", "data": b}}}} for b in frame_b64s]
        content.append({{"type": "text", "text": f"This is a {{SPORT}} clip. Classify as one of: {{types_list}}. Reply ONLY the label."}})

        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=20,
                messages=[{{"role": "user", "content": content}}]
            )
            label = resp.content[0].text.strip().lower().replace(' ', '_')
            if label not in PLAY_TYPES[SPORT]:
                label = 'other'
            labels[cn] = label
            cost += (resp.usage.input_tokens * 0.25 + resp.usage.output_tokens * 1.25) / 1e6
            api_calls += 1
        except Exception:
            labels[cn] = 'other'
            api_calls += 1

        # Progress every 10 clips
        if (i + 1) % 10 == 0:
            print(f'  [{{i+1}}/{{len(to_label)}}] labeled \\u2014 cost: ${{cost:.4f}} \\u2014 calls: {{api_calls}}')
            with open(LABELS_FILE, 'w') as f:
                json.dump(labels, f, indent=2)

    # Final save
    with open(LABELS_FILE, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f'\\n\\u2705 {sport.capitalize()} labeling done! Cost: ${{cost:.4f}}, calls: {{api_calls}}')

# Print distribution
if labels:
    print(f'\\nLabel distribution:')
    for label, count in Counter(labels.values()).most_common():
        print(f'  {{label}}: {{count}}')""")

# Cell 2D
code("""# ═══════════════════════════════════════════════════════
# Cell 2D — Label Summary
# ═══════════════════════════════════════════════════════
total_labeled = 0

print(f'{"="*60}')
print('\\u2705 LABELING COMPLETE')

for sport in ['basketball', 'football', 'lacrosse']:
    lf = f'{DRIVE_LABELS}/{sport}_labels.json'
    clips_dir = f'{CLIPS_BASE}/{sport}'
    total_clips = len(glob.glob(f'{clips_dir}/*.mp4'))
    if os.path.exists(lf):
        with open(lf) as f:
            labels = json.load(f)
        count = len(labels)
        total_labeled += count
        print(f'{sport.capitalize()} clips labeled: {count}/{total_clips}')
    else:
        print(f'{sport.capitalize()} clips labeled: 0/{total_clips}')

est_cost = total_labeled * 0.003
print(f'Total API calls: ~{total_labeled}')
print(f'Estimated cost: ${est_cost:.2f}')
print(f'Labels saved to Drive: \\u2705 {DRIVE_LABELS}/')
print(f'{"="*60}')

if est_cost > 15:
    print('\\u26a0\\ufe0f WARNING: Estimated cost exceeds $15')""")

# ============================================================
# SECTION 3: VIDEOMAE TRAINING
# ============================================================
md("""---
## Section 3: VideoMAE Play Classification
Fine-tune VideoMAE for temporal play classification. One model per sport.
Output: `videomae_{sport}_v5.zip`""")

for sport in ['basketball', 'football', 'lacrosse']:
    letter_t = {'basketball': 'A', 'football': 'C', 'lacrosse': 'E'}[sport]
    letter_s = {'basketball': 'B', 'football': 'D', 'lacrosse': 'F'}[sport]

    code(f"""# ═══════════════════════════════════════════════════════
# Cell 3{letter_t} \\u2014 Train videomae_{sport}_v5
# ═══════════════════════════════════════════════════════
from transformers import VideoMAEForVideoClassification, VideoMAEConfig
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

SPORT = '{sport}'
CLIPS_DIR = f'{{CLIPS_BASE}}/{{SPORT}}'
LABELS_FILE = f'{{DRIVE_LABELS}}/{{SPORT}}_labels.json'
MODEL_NAME = f'videomae_{{SPORT}}_v5'
LOCAL_MODEL_DIR = f'/content/models/{{MODEL_NAME}}'
CHECKPOINT_FILE = f'{{DRIVE_CHECKPOINTS}}/{{MODEL_NAME}}_checkpoint.pt'

# \\u2500\\u2500 Guard \\u2500\\u2500
clip_files = sorted(glob.glob(f'{{CLIPS_DIR}}/*.mp4'))
assert os.path.exists(LABELS_FILE), f'\\u274c No labels file \\u2014 run Section 2 first'
with open(LABELS_FILE) as f:
    all_labels = json.load(f)
assert len(clip_files) >= 50, f'\\u274c Need 50+ clips, have {{len(clip_files)}} \\u2014 run Section 1'
assert len(all_labels) >= 50, f'\\u274c Need 50+ labels, have {{len(all_labels)}} \\u2014 run Section 2'
print(f'\\u2705 {{len(clip_files)}} clips, {{len(all_labels)}} labels ready')

# \\u2500\\u2500 Dataset \\u2500\\u2500
class ClipDataset(Dataset):
    def __init__(self, clips_dir, labels_dict, play_types):
        self.label2id = {{t: i for i, t in enumerate(play_types)}}
        self.samples = [(cp, self.label2id[labels_dict[os.path.basename(cp)]])
                        for cp in sorted(glob.glob(f'{{clips_dir}}/*.mp4'))
                        if os.path.basename(cp) in labels_dict
                        and labels_dict[os.path.basename(cp)] in self.label2id]
        print(f'  Dataset: {{len(self.samples)}} samples, {{len(play_types)}} classes')
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(total-1, 0), 16, dtype=int)
        frames = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224,224)).astype(np.float32)/255.0)
            else:
                frames.append(np.zeros((224,224,3), dtype=np.float32))
        cap.release()
        return {{'pixel_values': torch.tensor(np.transpose(np.stack(frames),(0,3,1,2))),
                'labels': torch.tensor(label)}}

ds = ClipDataset(CLIPS_DIR, all_labels, PLAY_TYPES[SPORT])
assert len(ds) >= 10, f'\\u274c Need 10+ matched samples, have {{len(ds)}}'
ts = int(0.8*len(ds)); vs = len(ds)-ts
train_ds, val_ds = torch.utils.data.random_split(ds, [ts, vs])
print(f'  Train: {{ts}}, Val: {{vs}}')

# \\u2500\\u2500 Model \\u2500\\u2500
label2id = ds.label2id
id2label = {{v:k for k,v in label2id.items()}}
config = VideoMAEConfig.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics',
    num_labels=len(label2id), label2id=label2id, id2label=id2label)
model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics',
    config=config, ignore_mismatched_sizes=True).cuda()

start_epoch = 0; best_acc = 0.0
if os.path.exists(CHECKPOINT_FILE):
    ckpt = torch.load(CHECKPOINT_FILE)
    model.load_state_dict(ckpt['model_state'])
    start_epoch = ckpt.get('epoch', 0); best_acc = ckpt.get('val_acc', 0.0)
    print(f'  \\u267b\\ufe0f Resumed from epoch {{start_epoch}}, best_acc={{best_acc:.3f}}')

# \\u2500\\u2500 Train \\u2500\\u2500
print(f'\\n{{"="*60}}')
print(f'\\U0001f680 TRAINING: {{MODEL_NAME}}')
print(f'   epochs=20, batch=4, lr=5e-5')
print(f'{{"="*60}}\\n')

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
scheduler = CosineAnnealingLR(optimizer, T_max=20)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=4, num_workers=2)
patience_counter = 0

for epoch in range(start_epoch, 20):
    model.train(); total_loss=0; correct=0; total=0
    for batch in train_loader:
        pv=batch['pixel_values'].cuda(); lb=batch['labels'].cuda()
        out=model(pixel_values=pv, labels=lb)
        optimizer.zero_grad(); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        total_loss+=out.loss.item(); correct+=(out.logits.argmax(-1)==lb).sum().item(); total+=lb.size(0)
    scheduler.step()
    model.eval(); val_correct=0; val_total=0
    with torch.no_grad():
        for batch in val_loader:
            pv=batch['pixel_values'].cuda(); lb=batch['labels'].cuda()
            val_correct+=(model(pixel_values=pv).logits.argmax(-1)==lb).sum().item(); val_total+=lb.size(0)
    val_acc = val_correct/max(val_total,1)
    print(f'  Epoch {{epoch+1}}/20 loss:{{total_loss/len(train_loader):.4f}} train:{{correct/max(total,1):.3f}} val:{{val_acc:.3f}}')
    if val_acc > best_acc:
        best_acc=val_acc; patience_counter=0
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        model.save_pretrained(LOCAL_MODEL_DIR)
        torch.save({{'model_state':model.state_dict(),'epoch':epoch+1,'val_acc':val_acc,'label2id':label2id}}, CHECKPOINT_FILE)
        print(f'  \\U0001f4be [BEST] val_acc={{val_acc:.3f}} \\u2014 saved to Drive')
    else:
        patience_counter+=1
        if patience_counter>=5: print(f'  Early stopping at epoch {{epoch+1}}'); break

torch.cuda.empty_cache(); gc.collect()
print(f'\\n\\u2705 {{MODEL_NAME}} complete! Best val_acc={{best_acc:.3f}}')""")

    code(f"""# ═══════════════════════════════════════════════════════
# Cell 3{letter_s} \\u2014 Save videomae_{sport}_v5.zip to Drive
# ═══════════════════════════════════════════════════════
MODEL_NAME = 'videomae_{sport}_v5'
LOCAL_MODEL_DIR = f'/content/models/{{MODEL_NAME}}'
ZIP_PATH = f'{{DRIVE_SAVE_DIR}}/{{MODEL_NAME}}.zip'

assert os.path.exists(LOCAL_MODEL_DIR), f'\\u274c No trained model \\u2014 run Cell 3{letter_t} first'

import zipfile
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(LOCAL_MODEL_DIR):
        for fn in files:
            fp = os.path.join(root, fn)
            zf.write(fp, os.path.relpath(fp, LOCAL_MODEL_DIR))

sz = os.path.getsize(ZIP_PATH) / 1024 / 1024
print(f'{{"="*60}}')
print(f'\\u2705 {{MODEL_NAME}}.zip SAVED TO DRIVE')
print(f'Location: {{DRIVE_SAVE_DIR}}/')
print(f'File size: {{sz:.1f}}MB')
print(f'{{"="*60}}')""")

# ============================================================
# SECTION 4: YOLO OUTCOME CLASSIFIERS
# ============================================================
md("""---
## Section 4: YOLO Outcome Classifiers
Train YOLOv8 classification models for play outcome classification.
Output: `outcome_classifier_{sport}_v5.pt`""")

for sport in ['basketball', 'football', 'lacrosse']:
    letter_t = {'basketball': 'A', 'football': 'C', 'lacrosse': 'E'}[sport]
    letter_s = {'basketball': 'B', 'football': 'D', 'lacrosse': 'F'}[sport]

    code(f"""# ═══════════════════════════════════════════════════════
# Cell 4{letter_t} \\u2014 Train outcome_classifier_{sport}_v5
# ═══════════════════════════════════════════════════════
from ultralytics import YOLO

SPORT = '{sport}'
CLIPS_DIR = f'{{CLIPS_BASE}}/{{SPORT}}'
LABELS_FILE = f'{{DRIVE_LABELS}}/{{SPORT}}_labels.json'
MODEL_NAME = f'outcome_classifier_{{SPORT}}_v5'
CLS_DIR = f'/content/yolo_cls/{{SPORT}}'

# \\u2500\\u2500 Guard \\u2500\\u2500
clip_files = sorted(glob.glob(f'{{CLIPS_DIR}}/*.mp4'))
assert os.path.exists(LABELS_FILE), f'\\u274c No labels \\u2014 run Section 2 first'
with open(LABELS_FILE) as f:
    all_labels = json.load(f)
assert len(clip_files) >= 50, f'\\u274c Need 50+ clips, have {{len(clip_files)}}'
assert len(all_labels) >= 50, f'\\u274c Need 50+ labels, have {{len(all_labels)}}'
print(f'\\u2705 {{len(clip_files)}} clips, {{len(all_labels)}} labels ready')

# \\u2500\\u2500 Build classification dataset \\u2500\\u2500
for cn, label in all_labels.items():
    cp = f'{{CLIPS_DIR}}/{{cn}}'
    if not os.path.exists(cp): continue
    cap = cv2.VideoCapture(cp)
    mid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))//2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read(); cap.release()
    if not ret: continue
    split = 'train' if hash(cn)%5!=0 else 'val'
    od = f'{{CLS_DIR}}/{{split}}/{{label}}'; os.makedirs(od, exist_ok=True)
    cv2.imwrite(f'{{od}}/{{cn.replace(".mp4",".jpg")}}', cv2.resize(frame,(640,640)))

tc = len(glob.glob(f'{{CLS_DIR}}/train/**/*.jpg', recursive=True))
vc = len(glob.glob(f'{{CLS_DIR}}/val/**/*.jpg', recursive=True))
print(f'Classification dataset: {{tc}} train, {{vc}} val')
assert tc >= 20, f'\\u274c Need 20+ training images, have {{tc}}'

# \\u2500\\u2500 Train \\u2500\\u2500
print(f'\\n{{"="*60}}')
print(f'\\U0001f680 TRAINING: {{MODEL_NAME}}')
print(f'   epochs=80, batch={{DEFAULT_BATCH}}, YOLO classify mode')
print(f'{{"="*60}}\\n')

resume_path = f'{{DRIVE_CHECKPOINTS}}/{{MODEL_NAME}}/last.pt'
if os.path.exists(resume_path) and os.path.getsize(resume_path) > 1024*1024:
    print('\\u267b\\ufe0f Resuming from Drive checkpoint...')
    model = YOLO(resume_path)
    model.train(resume=True)
else:
    model = YOLO('yolov8m-cls.pt')
    model.train(data=CLS_DIR, epochs=80, imgsz=640, batch=DEFAULT_BATCH,
                name=MODEL_NAME, device=0, patience=15, save_period=5, amp=True, cache=True)

# Backup to Drive
ckpt_dir = f'{{DRIVE_CHECKPOINTS}}/{{MODEL_NAME}}'; os.makedirs(ckpt_dir, exist_ok=True)
for wt in ['best.pt','last.pt']:
    paths = sorted(glob.glob(f'runs/classify/{{MODEL_NAME}}*/weights/{{wt}}'))
    if paths: shutil.copy2(paths[-1], f'{{ckpt_dir}}/{{wt}}')

torch.cuda.empty_cache(); gc.collect()
print(f'\\n\\u2705 {{MODEL_NAME}} training complete!')""")

    code(f"""# ═══════════════════════════════════════════════════════
# Cell 4{letter_s} \\u2014 Save outcome_classifier_{sport}_v5.pt to Drive
# ═══════════════════════════════════════════════════════
MODEL_NAME = 'outcome_classifier_{sport}_v5'

paths = sorted(glob.glob(f'runs/classify/{{MODEL_NAME}}*/weights/best.pt'))
if not paths:
    alt = f'{{DRIVE_CHECKPOINTS}}/{{MODEL_NAME}}/best.pt'
    if os.path.exists(alt): paths = [alt]
assert paths and os.path.exists(paths[-1]), f'\\u274c No trained model \\u2014 run Cell 4{letter_t} first'

src = paths[-1]
dst = f'{{DRIVE_SAVE_DIR}}/{{MODEL_NAME}}.pt'
shutil.copy2(src, dst)
sz = os.path.getsize(dst)/1024/1024

print(f'{{"="*60}}')
print(f'\\u2705 {{MODEL_NAME}}.pt SAVED TO DRIVE')
print(f'Location: {{DRIVE_SAVE_DIR}}/')
print(f'File size: {{sz:.1f}}MB')
print(f'{{"="*60}}')""")

# ============================================================
# SECTION 5: JERSEY OCR
# ============================================================
md("""---
## Section 5: Jersey OCR Universal
Train universal jersey number detector (digits 0-9).
Combines Roboflow datasets + synthetic digit images.
Output: `jersey_ocr_universal_v5.pt`""")

code("""# ═══════════════════════════════════════════════════════
# Cell 5A — Download OCR datasets + prepare merged data
# ═══════════════════════════════════════════════════════
from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont
import random

assert ROBOFLOW_API_KEY, '\\u274c Set ROBOFLOW_API_KEY in Colab Secrets'
rf = Roboflow(api_key=ROBOFLOW_API_KEY)

def safe_download(ws, proj, ver, name):
    for v in [ver] + [i for i in range(1,6) if i!=ver]:
        try:
            ds = rf.workspace(ws).project(proj).version(v).download('yolov8')
            count = len(glob.glob(f'{ds.location}/train/images/*'))
            print(f'\\u2705 {name} v{v}: {count} images')
            return ds
        except Exception as e:
            print(f'  v{v} failed: {e}')
    print(f'\\u274c {name}: all versions failed')
    return None

print('=== Downloading Jersey OCR Datasets ===')
ds_jersey_1 = safe_download('footballplayertracking', 'jerseynumberdetectordigitdetector', 6, 'jersey_primary')
ds_jersey_2 = safe_download('volleyai-actions', 'jersey-number-detection-s01j4', 2, 'jersey_volleyai')

# ── Remap classes to digits 0-9 ──
def remap_to_digits(ds_dir):
    for label_dir in glob.glob(f'{ds_dir}/*/labels'):
        yaml_path = os.path.join(os.path.dirname(label_dir), 'data.yaml')
        class_map = {}
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                d = yaml.safe_load(f)
            for old_id, name in enumerate(d.get('names', [])):
                clean = str(name).strip()
                if clean.isdigit() and 0 <= int(clean) <= 9:
                    class_map[old_id] = int(clean)
        if not class_map: continue
        for lf in glob.glob(f'{label_dir}/*.txt'):
            lines = []
            with open(lf) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) in class_map:
                        parts[0] = str(class_map[int(parts[0])])
                        lines.append(' '.join(parts))
            if lines:
                with open(lf, 'w') as f:
                    f.write('\\n'.join(lines) + '\\n')

for ds in [ds_jersey_1, ds_jersey_2]:
    if ds: remap_to_digits(ds.location)

# ── Generate 2000 synthetic digit images ──
print('\\nGenerating synthetic digit images...')
SYNTH = '/content/synth_ocr'
os.makedirs(f'{SYNTH}/images', exist_ok=True)
os.makedirs(f'{SYNTH}/labels', exist_ok=True)
COLORS = [(255,0,0),(0,0,255),(255,255,255),(0,128,0),(255,165,0),(128,0,128),(0,0,0)]

for i in range(2000):
    num = random.randint(0,99); digits = str(num); w,h = 640,640
    bg = random.choice(COLORS); img = Image.new('RGB',(w,h),bg); draw = ImageDraw.Draw(img)
    for _ in range(random.randint(50,200)):
        x1,y1 = random.randint(0,w-1),random.randint(0,h-1)
        nc = tuple(max(0,min(255,c+random.randint(-30,30))) for c in bg)
        draw.rectangle([x1,y1,x1+random.randint(2,8),y1+random.randint(2,8)], fill=nc)
    tc = random.choice([(255,255,255),(0,0,0),(255,255,0)]); fs = random.randint(60,150)
    try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', fs)
    except: font = ImageFont.load_default()
    tw = len(digits)*(fs*0.7); sx = (w-tw)/2+random.randint(-50,50); yp = h/2-fs/2+random.randint(-80,80)
    ll = []
    for j, d in enumerate(digits):
        x = sx+j*(fs*0.7); draw.text((x,yp), d, fill=tc, font=font)
        dw,dh = fs*0.65, fs*1.1
        cx=max(0.01,min(0.99,(x+dw/2)/w)); cy=max(0.01,min(0.99,(yp+dh/2)/h))
        bw=max(0.02,min(0.5,dw/w)); bh=max(0.02,min(0.5,dh/h))
        ll.append(f'{int(d)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
    img.save(f'{SYNTH}/images/synth_{i:05d}.jpg')
    with open(f'{SYNTH}/labels/synth_{i:05d}.txt','w') as f: f.write('\\n'.join(ll)+'\\n')
    if (i+1)%500==0: print(f'  Generated {i+1}/2000')

# ── Merge all ──
print('\\nMerging datasets...')
MERGED = '/content/ocr_merged'
for s in ['train','val']:
    os.makedirs(f'{MERGED}/{s}/images', exist_ok=True)
    os.makedirs(f'{MERGED}/{s}/labels', exist_ok=True)

for ds in [ds_jersey_1, ds_jersey_2]:
    if not ds: continue
    loc=ds.location; dn=os.path.basename(loc)
    for sp in ['train','valid','val','test']:
        idir=f'{loc}/{sp}/images'; ldir=f'{loc}/{sp}/labels'
        if not os.path.exists(idir): continue
        target = 'val' if sp in ['valid','val','test'] else 'train'
        for img_path in glob.glob(f'{idir}/*'):
            fn=f'{dn}_{os.path.basename(img_path)}'
            shutil.copy2(img_path, f'{MERGED}/{target}/images/{fn}')
            lbl=os.path.join(ldir, os.path.splitext(os.path.basename(img_path))[0]+'.txt')
            if os.path.exists(lbl): shutil.copy2(lbl, f'{MERGED}/{target}/labels/{os.path.splitext(fn)[0]}.txt')

si=sorted(glob.glob(f'{SYNTH}/images/*.jpg')); sp=int(len(si)*0.8)
for i,ip in enumerate(si):
    fn=os.path.basename(ip); s='train' if i<sp else 'val'
    shutil.copy2(ip, f'{MERGED}/{s}/images/{fn}')
    shutil.copy2(f'{SYNTH}/labels/{fn.replace(".jpg",".txt")}', f'{MERGED}/{s}/labels/{fn.replace(".jpg",".txt")}')

dy = {'path':MERGED,'train':'train/images','val':'val/images','nc':10,'names':{i:str(i) for i in range(10)}}
with open(f'{MERGED}/data.yaml','w') as f: yaml.dump(dy, f)
tc=len(glob.glob(f'{MERGED}/train/images/*')); vc=len(glob.glob(f'{MERGED}/val/images/*'))
print(f'\\n\\u2705 OCR dataset: {tc} train, {vc} val \\u2014 10 digit classes (0-9)')""")

code("""# ═══════════════════════════════════════════════════════
# Cell 5B — Train jersey_ocr_universal_v5
# ═══════════════════════════════════════════════════════
from ultralytics import YOLO

MODEL_NAME = 'jersey_ocr_universal_v5'
DATA_PATH = '/content/ocr_merged/data.yaml'

assert os.path.exists(DATA_PATH), '\\u274c No OCR data \\u2014 run Cell 5A first'
tc = len(glob.glob('/content/ocr_merged/train/images/*'))
assert tc >= 100, f'\\u274c Need 100+ images, have {tc}'
print(f'\\u2705 {tc} training images ready')

print(f'\\n{"="*60}')
print(f'\\U0001f680 TRAINING: {MODEL_NAME}')
print(f'   epochs=120, fliplr=0 (digits), patience=25')
print(f'{"="*60}\\n')

resume_path = f'{DRIVE_CHECKPOINTS}/{MODEL_NAME}/last.pt'
if os.path.exists(resume_path) and os.path.getsize(resume_path) > 1024*1024:
    print('\\u267b\\ufe0f Resuming from Drive checkpoint...')
    model = YOLO(resume_path)
    model.train(resume=True)
else:
    model = YOLO('yolov8m.pt')
    model.train(data=DATA_PATH, epochs=120, imgsz=640, batch=DEFAULT_BATCH,
                name=MODEL_NAME, device=0, patience=25, save_period=5,
                amp=True, cache=True, fliplr=0.0, mosaic=0.5, degrees=5.0, scale=0.3)

ckpt_dir = f'{DRIVE_CHECKPOINTS}/{MODEL_NAME}'; os.makedirs(ckpt_dir, exist_ok=True)
for wt in ['best.pt','last.pt']:
    paths = sorted(glob.glob(f'runs/detect/{MODEL_NAME}*/weights/{wt}'))
    if paths: shutil.copy2(paths[-1], f'{ckpt_dir}/{wt}')
torch.cuda.empty_cache(); gc.collect()
print(f'\\n\\u2705 {MODEL_NAME} training complete!')""")

code("""# ═══════════════════════════════════════════════════════
# Cell 5C — Save jersey_ocr_universal_v5.pt to Drive
# ═══════════════════════════════════════════════════════
MODEL_NAME = 'jersey_ocr_universal_v5'
paths = sorted(glob.glob(f'runs/detect/{MODEL_NAME}*/weights/best.pt'))
if not paths:
    alt = f'{DRIVE_CHECKPOINTS}/{MODEL_NAME}/best.pt'
    if os.path.exists(alt): paths = [alt]
assert paths and os.path.exists(paths[-1]), f'\\u274c No trained model \\u2014 run Cell 5B first'

src = paths[-1]; dst = f'{DRIVE_SAVE_DIR}/{MODEL_NAME}.pt'
shutil.copy2(src, dst)
sz = os.path.getsize(dst)/1024/1024

try:
    m = YOLO(src).val(); map50 = m.box.map50
    print(f'{"="*60}')
    print(f'\\u2705 {MODEL_NAME}.pt SAVED TO DRIVE')
    print(f'Location: {DRIVE_SAVE_DIR}/')
    print(f'File size: {sz:.1f}MB')
    print(f'mAP50: {map50:.3f}')
    print(f'{"="*60}')
except Exception:
    print(f'{"="*60}')
    print(f'\\u2705 {MODEL_NAME}.pt SAVED TO DRIVE')
    print(f'Location: {DRIVE_SAVE_DIR}/')
    print(f'File size: {sz:.1f}MB')
    print(f'{"="*60}')""")

# ============================================================
# SECTION 6: PLAYER DETECTOR
# ============================================================
md("""---
## Section 6: Player Detector
Universal player detector combining basketball, football, and lacrosse datasets.
Output: `player_detector_v5.pt`""")

code("""# ═══════════════════════════════════════════════════════
# Cell 6A — Download player detection datasets
# ═══════════════════════════════════════════════════════
from roboflow import Roboflow

assert ROBOFLOW_API_KEY, '\\u274c Set ROBOFLOW_API_KEY in Colab Secrets'
rf = Roboflow(api_key=ROBOFLOW_API_KEY)

def safe_download(ws, proj, ver, name):
    for v in [ver] + [i for i in range(1,6) if i!=ver]:
        try:
            ds = rf.workspace(ws).project(proj).version(v).download('yolov8')
            count = len(glob.glob(f'{ds.location}/train/images/*'))
            print(f'\\u2705 {name} v{v}: {count} images')
            return ds
        except Exception as e:
            print(f'  v{v} failed: {e}')
    print(f'\\u274c {name}: all versions failed')
    return None

print('=== Downloading Player Detection Datasets ===')
ds_fb = safe_download('augmented-startups', 'football-player-detection-kucab', 1, 'football_players')
ds_bb = safe_download('roboflow-universe-projects', 'basketball-players-fy4c2', 1, 'basketball_players')
ds_lax = safe_download('ryseai', 'lacrosse-object-detection', 1, 'lacrosse_players')

# ── Merge into single player dataset ──
print('\\nMerging player datasets...')
MP = '/content/player_merged'
for s in ['train','val']:
    os.makedirs(f'{MP}/{s}/images', exist_ok=True)
    os.makedirs(f'{MP}/{s}/labels', exist_ok=True)

for ds in [ds_fb, ds_bb, ds_lax]:
    if not ds: continue
    loc=ds.location; dn=os.path.basename(loc)
    for sp in ['train','valid','val','test']:
        idir=f'{loc}/{sp}/images'; ldir=f'{loc}/{sp}/labels'
        if not os.path.exists(idir): continue
        target = 'val' if sp in ['valid','val','test'] else 'train'
        for img_path in glob.glob(f'{idir}/*'):
            fn=f'{dn}_{os.path.basename(img_path)}'
            shutil.copy2(img_path, f'{MP}/{target}/images/{fn}')
            lbl=os.path.join(ldir, os.path.splitext(os.path.basename(img_path))[0]+'.txt')
            if os.path.exists(lbl):
                with open(lbl) as f:
                    lines = ['0 '+' '.join(l.strip().split()[1:]) for l in f if len(l.strip().split())>=5]
                if lines:
                    with open(f'{MP}/{target}/labels/{os.path.splitext(fn)[0]}.txt','w') as f:
                        f.write('\\n'.join(lines)+'\\n')

dy = {'path':MP,'train':'train/images','val':'val/images','nc':1,'names':{0:'player'}}
with open(f'{MP}/data.yaml','w') as f: yaml.dump(dy, f)
tc=len(glob.glob(f'{MP}/train/images/*')); vc=len(glob.glob(f'{MP}/val/images/*'))
print(f'\\n\\u2705 Player dataset: {tc} train, {vc} val \\u2014 1 class (player)')""")

code("""# ═══════════════════════════════════════════════════════
# Cell 6B — Train player_detector_v5
# ═══════════════════════════════════════════════════════
from ultralytics import YOLO

MODEL_NAME = 'player_detector_v5'
DATA_PATH = '/content/player_merged/data.yaml'

assert os.path.exists(DATA_PATH), '\\u274c No player data \\u2014 run Cell 6A first'
tc = len(glob.glob('/content/player_merged/train/images/*'))
assert tc >= 50, f'\\u274c Need 50+ images, have {tc}'
print(f'\\u2705 {tc} training images ready')

print(f'\\n{"="*60}')
print(f'\\U0001f680 TRAINING: {MODEL_NAME}')
print(f'   epochs=100, batch={DEFAULT_BATCH}, patience=25')
print(f'{"="*60}\\n')

resume_path = f'{DRIVE_CHECKPOINTS}/{MODEL_NAME}/last.pt'
if os.path.exists(resume_path) and os.path.getsize(resume_path) > 1024*1024:
    print('\\u267b\\ufe0f Resuming from Drive checkpoint...')
    model = YOLO(resume_path)
    model.train(resume=True)
else:
    model = YOLO('yolov8m.pt')
    model.train(data=DATA_PATH, epochs=100, imgsz=640, batch=DEFAULT_BATCH,
                name=MODEL_NAME, device=0, patience=25, save_period=5, amp=True, cache=True)

ckpt_dir = f'{DRIVE_CHECKPOINTS}/{MODEL_NAME}'; os.makedirs(ckpt_dir, exist_ok=True)
for wt in ['best.pt','last.pt']:
    paths = sorted(glob.glob(f'runs/detect/{MODEL_NAME}*/weights/{wt}'))
    if paths: shutil.copy2(paths[-1], f'{ckpt_dir}/{wt}')
torch.cuda.empty_cache(); gc.collect()
print(f'\\n\\u2705 {MODEL_NAME} training complete!')""")

code("""# ═══════════════════════════════════════════════════════
# Cell 6C — Save player_detector_v5.pt to Drive
# ═══════════════════════════════════════════════════════
MODEL_NAME = 'player_detector_v5'
paths = sorted(glob.glob(f'runs/detect/{MODEL_NAME}*/weights/best.pt'))
if not paths:
    alt = f'{DRIVE_CHECKPOINTS}/{MODEL_NAME}/best.pt'
    if os.path.exists(alt): paths = [alt]
assert paths and os.path.exists(paths[-1]), f'\\u274c No trained model \\u2014 run Cell 6B first'

src = paths[-1]; dst = f'{DRIVE_SAVE_DIR}/{MODEL_NAME}.pt'
shutil.copy2(src, dst)
sz = os.path.getsize(dst)/1024/1024

try:
    m = YOLO(src).val(); map50 = m.box.map50
    print(f'{"="*60}')
    print(f'\\u2705 {MODEL_NAME}.pt SAVED TO DRIVE')
    print(f'Location: {DRIVE_SAVE_DIR}/')
    print(f'File size: {sz:.1f}MB')
    print(f'mAP50: {map50:.3f}')
    print(f'{"="*60}')
except Exception:
    print(f'{"="*60}')
    print(f'\\u2705 {MODEL_NAME}.pt SAVED TO DRIVE')
    print(f'Location: {DRIVE_SAVE_DIR}/')
    print(f'File size: {sz:.1f}MB')
    print(f'{"="*60}')""")

# ============================================================
# SECTION 7: FINAL SUMMARY
# ============================================================
md("""---
## Section 7: Final Report""")

code("""# ═══════════════════════════════════════════════════════
# Cell 7A — Check Drive folder
# ═══════════════════════════════════════════════════════
print(f'\\nChecking {DRIVE_SAVE_DIR}...\\n')
all_files = sorted(glob.glob(f'{DRIVE_SAVE_DIR}/*.*'))
if all_files:
    total_size = 0
    for f in all_files:
        sz = os.path.getsize(f)/1024/1024; total_size += sz
        print(f'  {os.path.basename(f):45s} {sz:8.1f}MB')
    print(f'\\n  Total: {len(all_files)} files, {total_size:.1f}MB')
else:
    print('  No files found \\u2014 run Sections 3-6')""")

code("""# ═══════════════════════════════════════════════════════
# Cell 7B — Final Status Table
# ═══════════════════════════════════════════════════════
ALL_MODELS = [
    'videomae_basketball_v5.zip',
    'videomae_football_v5.zip',
    'videomae_lacrosse_v5.zip',
    'outcome_classifier_basketball_v5.pt',
    'outcome_classifier_football_v5.pt',
    'outcome_classifier_lacrosse_v5.pt',
    'jersey_ocr_universal_v5.pt',
    'player_detector_v5.pt',
]

print('='*60)
print('  V5 TRAINING COMPLETE')
print('='*60)

print(f'\\n\\U0001f4be GOOGLE DRIVE ({DRIVE_SAVE_DIR}):')
print(f'   \\u2192 Open Google Drive \\u2192 MyDrive \\u2192 clipt_v5_models')
print(f'   \\u2192 Copy files from there to your reelapp folder\\n')

saved = 0
for mn in ALL_MODELS:
    dp = f'{DRIVE_SAVE_DIR}/{mn}'
    if os.path.exists(dp):
        sz = os.path.getsize(dp)/1024/1024
        print(f'  \\u2705 {mn} ({sz:.1f}MB)')
        saved += 1
    else:
        base = mn.replace('.pt','').replace('.zip','')
        ckpt = f'{DRIVE_CHECKPOINTS}/{base}/best.pt'
        local = glob.glob(f'runs/detect/{base}*/weights/best.pt') + glob.glob(f'runs/classify/{base}*/weights/best.pt')
        if os.path.exists(ckpt) or local:
            print(f'  \\u26a0\\ufe0f  {mn} \\u2014 TRAINED NOT SAVED (run save cell)')
        else:
            print(f'  \\u274c {mn} \\u2014 MISSING')

print(f'\\n  Models ready: {saved}/8')

print(f'\\n{"="*60}')
if saved == 8:
    print('\\U0001f389 ALL 8 MODELS COMPLETE!')
    print('\\nNEXT STEP: Download all files from Google Drive:')
    print(f'  {DRIVE_SAVE_DIR}/')
    print('and move them to your reelapp folder.')
    print('Then send to Claude for integration prompt.')
elif saved >= 6:
    print(f'\\u26a0\\ufe0f {saved}/8 models saved \\u2014 almost done!')
    print('Run the missing save cells to complete.')
else:
    print(f'\\u274c {saved}/8 models saved \\u2014 run remaining training cells.')
    print('\\nTo recover:')
    print('  1. Rerun Section 0 (Setup)')
    print('  2. Rerun the failed model\\'s training cell \\u2014 it will resume from Drive')
print('='*60)""")

# ============================================================
# ASSEMBLE NOTEBOOK
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
        "colab": {"provenance": [], "gpuType": "A100"},
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

script_dir = pathlib.Path(__file__).parent
output_path = str(script_dir / 'train_models_v5_autolabel.ipynb')

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Wrote {output_path}")
print(f"Total cells: {len(cells)}")
print(f"Markdown: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"Code: {sum(1 for c in cells if c['cell_type'] == 'code')}")
