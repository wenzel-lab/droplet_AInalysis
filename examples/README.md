# Droplet AInalysis – Examples Folder

This folder provides a complete, ready-to-run example suite for the droplet detection and analysis pipeline. It is intended for users who want to quickly test the functionality, explore sample data, and understand the workflow without extra setup.

## 📦 Contents

- **PARAMETERS.py**: Example configuration for running the scripts in this folder.
- **data_management/**: Core helper modules for droplet statistics, annotation, and data aggregation.
- **imgs/real_imgs/**: Sample images (PNG, JPG) for testing detection and analysis.
- **imgs/results/**: Output directory where results (annotated images, plots, etc.) are saved after running analysis scripts.
- **weights/**: Pretrained YOLOv8 model weights for immediate use.
- **see_ellipses.py**: Script to visualize detected droplet ellipses on sample images.
- **see_distributions.py**: Script to plot droplet size/volume distributions from sample images.

## 🚀 Quick Start

1. **Install dependencies:**
   Use the main repository's `packages.py` or `requirements.txt` to install all required packages.
   ```bash
   python ../packages.py
   # or
   pip install -r ../requirements.txt
   ```

2. **Run an example script:**
   - To visualize droplet ellipses:
     ```bash
     python see_ellipses.py
     ```
   - To plot droplet distributions:
     ```bash
     python see_distributions.py
     ```

3. **View results:**
   - Outputs (annotated images, plots, CSVs) will appear in `imgs/results/`.

## 📝 Notes

- **No extra setup required:** All sample data and weights are included for immediate experimentation.
- **Custom images:** Place your own images in `imgs/real_imgs/` to analyze them with the example scripts.
- **Model weights:** You can swap in your own YOLOv8 `.pt` files by placing them in the `weights/` directory and updating `PARAMETERS.py`.
- **Output cleanup:** You may safely delete contents of `imgs/results/` to clear previous outputs.

## 📚 More Information

For full documentation, troubleshooting, and advanced usage, see the main repository `README.md`.

---
