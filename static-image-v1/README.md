# Droplet AInalysis – Static Image Version (v1)

This tool performs automated detection and analysis of droplets in static microscopy images using deep learning (YOLOv8). It generates annotated images, droplet size/volume statistics, and visual summaries (including Gaussian fits). Designed for researchers working with droplet microfluidics or similar applications.

---

## ✅ Features

| Feature                        | Description |
|--------------------------------|-------------|
| 📷 **Static image input**      | Analyze single or batch images (`.png`, `.jpg`) |
| 🧠 **YOLOv8-based detection**  | Uses Ultralytics models (`.pt`) for droplet identification |
| 🥇 **Model selection**         | Automatically compares weights and selects the best model |
| 📊 **Enhanced statistics**     | Mean, standard deviation, mode & coefficient of variation (CV) displayed on plots |
| 📈 **Multiple plot types**     | Generates mixed (hist+fit), histogram-only & normal-fit-only plots for diameter & volume |
| 🗂 **Organized outputs**       | Separates `predictions/` and `statistics/` folders; includes `individual_weights/` previews |
| 📁 **CSV export**              | Saves per-droplet and summary statistics |
| 🎞️ **GIF of model iterations**| Generates a visual history of model predictions |

---

## 📦 Installation & Requirements

- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (`pip install ultralytics`)
- numpy, pandas, matplotlib, seaborn, opencv-python

Install requirements:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

1. Place your images in the `imgs/real_imgs/` directory.
2. Place YOLOv8 model weights (`.pt` files) in the `weights/` directory.
3. Run the analysis script:
    ```bash
    python main.py
    ```
Then choose:
- Whether to analyze a single test image (`PARAMETERS.py`) or all images
- Whether to use the predefined YOLO model or test all .pt models and select the best by score

4. Results will be saved in `imgs/results/<image_name>/predictions/` and `imgs/results/<image_name>/statistics/`

---

## 📂 Outputs (per image)

| File / Folder                                   | Purpose                                  |
|-------------------------------------------------|------------------------------------------|
| `predictions/best_prediction.jpg`               | Annotated image with overlays            |
| `predictions/individual_weights/*.jpg`          | Preview of each weight's predictions     |
| `predictions/history.gif`                       | GIF of predictions per model             |
| `predictions/model_performance.png`             | Model comparison chart                   |
| `statistics/diameter_mixed.png`                 | Diameter histogram + normal fit          |
| `statistics/diameter_hist.png`                  | Diameter histogram only                  |
| `statistics/diameter_normal.png`                | Diameter normal fit (density)            |
| `statistics/volume_mixed.png`                   | Volume histogram + normal fit            |
| `statistics/volume_hist.png`                    | Volume histogram only                    |
| `statistics/volume_normal.png`                  | Volume normal fit (density)              |
| `statistics/droplet_measurements.csv`           | Droplet-level data and summary           |

---

## 🖼 Example

Example input images and output files can be found in `imgs/results/droplets/`.

INPUT:
![Example Input](imgs/real_imgs/droplets.png)

OUTPUT:
![Example Output](imgs/results/droplets/best_prediction.jpg)

![Example Output](imgs/results/droplets/history.gif)

---

## Configure Parameters

Adjust `PARAMETERS.py` to control behavior:

```bash
TEST_IMAGE = "droplets.png"
WEIGHT = "best_model.pt"
IMGSZ = 1024
CONFIDENCE = 0.75
MAX_DETECT = 500
PIXEL_RATIO = 1.0
UNIT = "µm"
```