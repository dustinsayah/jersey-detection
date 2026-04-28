# Colab Notebook Design — Real-ESRGAN Video Upscaling

**Status:** design / research — not yet implemented as `.ipynb`.

## Goal

Take Dustin's 360p game film (cached at `C:/Users/dusti/clipt-test/source.mp4` or
the YouTube source) and produce a 720p version that Railway's detection
pipeline can consume. Higher input resolution should improve EasyOCR jersey
hits (currently ~8% at 360p) and tighten the frame-diff motion signal we
recorded as `realMotionAvg` in v8.32.0 but did not yet enable as a filter.

## Key research findings driving the design

- **Best model for sports / non-anime footage:** `RealESRGAN_x4plus` (general
  realistic model, ~4× scale). For faster runs the small companion model
  `realesr-general-x4v3` works but trades quality.
- **Upscale ratio:** we want 2× (360p → 720p), not 4× (1440p has diminishing
  returns and 4× the inference cost). Use `--outscale 2` to clamp.
- **A100 throughput:** documented inference fluctuates widely; published
  Replicate runs of `lucataco/real-esrgan-video` average ~6 minutes for short
  clips. No reliable A100 fps benchmark in the public docs, so the notebook
  should print per-batch timing on the first run and let us calibrate.
- **Strategy choice:** only upscale **detected clip windows**, not the entire
  2-hour game. 8 clips × 15 s × 30 fps ≈ 3,600 frames. That's the right scope.

## Notebook outline (cells)

### Cell 1 — Mount Drive + verify GPU

```python
from google.colab import drive
drive.mount('/content/drive')

import torch
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))
# Expect: CUDA: True  NVIDIA A100-SXM4-40GB
```

### Cell 2 — Install dependencies

```bash
%%bash
pip install -q basicsr facexlib gfpgan
pip install -q realesrgan
git clone https://github.com/xinntao/Real-ESRGAN.git /content/Real-ESRGAN
cd /content/Real-ESRGAN && pip install -q -r requirements.txt && python setup.py develop -q
# Pre-download weights so the inference script doesn't stall on first call
wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
     -O /content/Real-ESRGAN/weights/RealESRGAN_x4plus.pth
```

### Cell 3 — Inputs

Two modes. Mode A is what we'll use first:

**Mode A — already-extracted clips.** Detection emits `clips[]` with
`startTime`/`endTime`. Upload a JSON manifest to Drive:

```json
{
  "source_url": "https://res.cloudinary.com/.../source.mp4",
  "clips": [
    {"id": 1, "startTime": 1683.0, "endTime": 1701.0},
    {"id": 2, "startTime": 4410.0, "endTime": 4425.0}
  ]
}
```

**Mode B — entire video.** Same code path but treat the full file as one clip.

```python
import json, pathlib
manifest = json.loads(pathlib.Path('/content/drive/MyDrive/clipt/manifest.json').read_text())
SRC = manifest['source_url']
CLIPS = manifest['clips']
```

### Cell 4 — Download source + cut clips with ffmpeg

```bash
%%bash
mkdir -p /content/work/segments /content/work/upscaled
# yt-dlp handles both YouTube URLs and direct mp4 (HTTP)
pip install -q yt-dlp
yt-dlp -f 'best[ext=mp4]/best' -o /content/work/source.mp4 "$SRC" 2>/dev/null || \
    curl -L -o /content/work/source.mp4 "$SRC"
ffprobe -v quiet -print_format json -show_streams /content/work/source.mp4 | \
    python -c "import json,sys; s=json.load(sys.stdin)['streams']; \
print([(x.get('width'), x.get('height')) for x in s if x.get('codec_type')=='video'])"
```

```python
import subprocess
for c in CLIPS:
    out = f"/content/work/segments/clip_{c['id']:02d}.mp4"
    subprocess.run([
        'ffmpeg', '-y', '-ss', str(c['startTime']),
        '-to', str(c['endTime']),
        '-i', '/content/work/source.mp4',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-c:a', 'aac', out,
    ], check=True, capture_output=True)
print('Cut', len(CLIPS), 'segments')
```

### Cell 5 — Upscale each clip with `inference_realesrgan_video.py`

```python
import time, subprocess, glob

t0 = time.perf_counter()
for seg in sorted(glob.glob('/content/work/segments/clip_*.mp4')):
    out = seg.replace('/segments/', '/upscaled/')
    subprocess.run([
        'python', '/content/Real-ESRGAN/inference_realesrgan_video.py',
        '-i', seg, '-o', '/content/work/upscaled',
        '-n', 'RealESRGAN_x4plus',
        '--outscale', '2',     # 360p → 720p (model is natively 4×)
        '--num_process_per_gpu', '2',
        '--fp32',              # tradeoff: A100 has tensor cores; fp32 is safer for sharp text
    ], check=True)
print(f'Upscaled {len(CLIPS)} clips in {time.perf_counter()-t0:.1f}s')
```

**Expected timing on A100 (estimate, calibrate on first run):**
~30–80 fps for a 360p×4 inference. 3,600 frames / ~50 fps ≈ 72 s wall time
plus per-clip ffmpeg overhead. Conservative budget: **5–10 minutes for ~8 clips**.

### Cell 6 — Upload to Cloudinary so Railway can pull

```python
import cloudinary, cloudinary.uploader, os
cloudinary.config(
    cloud_name=os.environ['CLD_NAME'],
    api_key=os.environ['CLD_KEY'],
    api_secret=os.environ['CLD_SECRET'],
)
results = []
for f in sorted(glob.glob('/content/work/upscaled/*.mp4')):
    r = cloudinary.uploader.upload_large(
        f, resource_type='video',
        public_id=f"clipt/upscaled/{os.path.basename(f).split('.')[0]}",
        overwrite=True,
    )
    results.append({'clip': os.path.basename(f), 'url': r['secure_url']})
print(json.dumps(results, indent=2))
```

The returned URLs go into `clips[].upscaledUrl` on the Railway side, and the
render server pulls those instead of the originals.

## Connecting to Railway

Two options:

1. **Manual:** copy the JSON of upscaled URLs into the existing detection
   result and re-run extract-and-render. Lowest risk while we validate.
2. **Webhook:** add a route on the detection server (e.g. `/upscale-callback`)
   that accepts `{job_id, clips: [{id, url}]}` and updates the in-memory job
   record. Cleaner, but defer until the manual path is proven.

## Open questions before implementing

- Does the upscaled output preserve enough sharpness on jersey numbers for
  EasyOCR/PARSeq to actually hit? Validate on **one** real clip end-to-end
  before scaling out.
- Is upscaling worth it if we move to PARSeq fine-tuned weights? Possibly the
  PARSeq notebook (`docs/colab_parseq_training.md`) makes this redundant.
- Do we want pre-snap frames or full-clip upscaled output? Probably full clip,
  since the current bottleneck is OCR at action time, not pre-snap.
