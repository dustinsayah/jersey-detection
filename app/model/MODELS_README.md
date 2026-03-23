# Clipt Detection Models

## Ali's Models (existing, committed)
- `jersey_number_yolo11m.pt` — YOLO11M jersey detection (40.6 MB)
- `yolo26n-seg.pt` — Person segmentation (downloaded in Dockerfile)
- `yolo11n-pose.pt` — Pose estimation (downloaded in Dockerfile)
- `yamnet.tflite` — Audio classification (downloaded in Dockerfile)
- `public/` — Uncertainty-JNR + ParseQ checkpoints (bootstrapped in Dockerfile)

## Roboflow Trained Models (need Colab training first)

Run `notebooks/train_models.ipynb` on Google Colab (free T4 GPU), then place the downloaded `.pt` files here.

| File | Source Dataset | Images | Purpose |
|------|---------------|--------|---------|
| `football_digit_detector.pt` | FootballPlayerTracking/jerseynumberdetectordigitdetector | 13,815 | Primary football digit OCR |
| `football_player_detector.pt` | roboflow-jvuqo/football-players-detection-3zvbc | 372 | Player bounding boxes (crop → OCR) |
| `basketball_jersey_ocr.pt` | roboflow-jvuqo/basketball-jersey-numbers-ocr | ~200 | Basketball jersey numbers |
| `football_jersey_tracker.pt` | football-tracking/football-jersey-tracker | ~300 | Football jersey tracking |

Training takes ~2 hours total on Colab free tier (T4 GPU).
Expected mAP50: >0.7 for digit detection, >0.85 for player detection.

## After adding .pt files

```bash
git add app/model/*.pt
git commit -m "Add Roboflow trained detection models"
git push
```

Railway will auto-redeploy. Check `/health` endpoint — `roboflow_models` should show all 4 as `"loaded"`.
