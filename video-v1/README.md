# Video Analysis - v1

Automated batch analysis of droplet videos using deep learning, with output and statistics.

---

## Overview

This module provides robust video analysis for droplet detection and measurement. It processes videos in batch mode, applies YOLO-based detection to each frame, and outputs annotated videos, per-frame images, measurement CSVs, and summary/statistics plots.

---

## ✅ Features

| Feature | Description |
|---------|-------------|
| 🎞️ **Batch video input** | Analyze single or multiple videos in `videos/samples/` |
| 🧠 **YOLOv8-based detection** | Uses Ultralytics models (`.pt`) for droplet identification |
| ⚙️ **Interactive frame selection** | Choose how many frames to analyze per video at runtime |
| 🏷️ **Enhanced statistics** | Mean, std-dev, mode & CV, plus Gaussian fits for diameter & volume |
| 🗂 **Organized outputs** | Separate `predictions/` (video & frames) and `statistics/` (plots & CSV) |
| 📈 **Multiple plot types** | Mixed (hist+fit), histogram-only & normal-fit-only for both metrics |
| 📁 **CSV export** | Saves per-droplet and summary statistics |


---

## Folder Structure
```
video-v1/
├── main.py
├── PARAMETERS.py
├── README.md
├── data_management/
├── ...
├── videos/
│   ├── samples/         # Place your input videos here
│   └── results/         # Output folders for each analyzed video
```
---

## Usage
1. **Configure parameters:**
   - Edit `PARAMETERS.py` to set analysis parameters, e.g.:
     - `IMGSZ`, `CONFIDENCE`, `PIXEL_RATIO`, `WEIGHT`, etc.
     - To analyze a single video, set `VIDEO_TO_ANALYZE = "your_video.mp4"`.
     - To analyze all videos, leave `VIDEO_TO_ANALYZE = ""`.
2. **Run analysis:**
   ```bash
   python main.py
   ```
3. **Select frame count:**
   - When prompted, enter the number of frames to analyze per video (or press Enter for all).
4. **Review results:**
   - Output for each video will be in `videos/results/<video_name>/`:
     - `annotated_<video_name>.mp4` — Annotated output video
     - `annotated_frames/` — Per-frame annotated images
     - `droplet_measurements.csv` — Measurements with summary
     - `video_analysis_stats.txt` — Frame and droplet summary
     - `diameter_hist.png`, `volume_hist.png` — Distribution plots

---

## PARAMETERS.py Reference
- `IMGSZ` — Model input size (should match your training)
- `CONFIDENCE` — Minimum confidence for droplet detection
- `PIXEL_RATIO` — Pixel-to-micron conversion
- `WEIGHT` — Path to YOLO weights
- `VIDEO_TO_ANALYZE` — Set to filename for single video, or leave blank for all

---

## 📂 Outputs (per video)

| File / Folder | Purpose |
|---------------|---------|
| `predictions/annotated_<name>.mp4` | Annotated output video |
| `predictions/annotated_frames/*.jpg` | Per-frame annotated images |
| `statistics/droplet_measurements.csv` | Droplet-level data and summary |
| `statistics/diameter_mixed.png` | Diameter histogram + normal fit |
| `statistics/diameter_hist.png` | Diameter histogram only |
| `statistics/diameter_normal.png` | Diameter normal fit (density) |
| `statistics/volume_mixed.png` | Volume histogram + normal fit |
| `statistics/volume_hist.png` | Volume histogram only |
| `statistics/volume_normal.png` | Volume normal fit (density) |
| `statistics/video_analysis_stats.txt` | Frame & droplet summary |


---

## Troubleshooting
- **No output / empty results:**
  - Check video format and integrity.
  - Ensure YOLO weights are correct and compatible.
- **Progress bar but no detections:**
  - Try lowering confidence or adjusting model parameters.
- **Logging errors:**
  - Logging and print output are suppressed for clean progress. If you need debug info, comment out the suppression in `main.py`.

---

## See Also
- [static-image-v1/README.md](../static-image-v1/README.md)
- [webcam-v1/README.md](../webcam-v1/README.md)

---

For questions or issues, open an issue in this repository.
