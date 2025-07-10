# Training Field

This folder contains all resources for generating, preparing, and training YOLO models for droplet detection in the Droplet AInalysis project.

## 📁 Folder Structure

- `datasets/` — Contains training and validation datasets (images and labels).
- `real_samples/` — Real microscopy images for validation or augmentation.
- `runs/` — Output directory for training results (trained models, logs, metrics).
- `image_generator.py` — Script for generating synthetic training images and annotations.
- `image_tools.py` — Utilities for image processing and annotation.
- `make_training_data.py` — Prepares datasets in YOLO format for training.
- `train.py` — Main script to launch YOLO model training.
- `train_1.yaml` — Example YOLO training configuration file.

## 🚀 Usage

1. **Generate Synthetic Data:**
   ```bash
   python image_generator.py
   ```
2. **Prepare Training Data:**
   ```bash
   python make_training_data.py
   ```
3. **Train the Model:**
   ```bash
   python train.py --config train_1.yaml
   ```
   - Training results and logs are saved in `runs/`.

## 🛠️ Extending & Modifying

- Add new datasets to `datasets/` or `real_samples/`.
- Add or modify training configurations as `.yaml` files.
- Document new scripts with clear docstrings and comments.
- To add new data augmentation or preprocessing steps, update `image_tools.py` or `image_generator.py`.

## 📦 Dependencies

- Python 3.8+
- Ultralytics YOLO (https://github.com/ultralytics/ultralytics)
- OpenCV
- numpy
- (Other scientific Python libraries as needed)

## 🤝 Collaboration Tips

- Keep the `runs/` directory for reproducibility and tracking experiments.
- Use clear, descriptive names for new scripts, datasets, and configs.
- Add docstrings and comments to all new functions and scripts.
- If you add new scripts or major features, update this README.

## 📝 Notes

- Large datasets or output files can be gitignored if not needed for version control.
- For questions or contributions, see the main project README for contact info and guidelines.

---

*This README is intended to help both new and returning developers understand and extend the training pipeline efficiently.*
