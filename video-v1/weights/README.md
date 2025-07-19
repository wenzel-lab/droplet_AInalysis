# Weights Folder

This directory stores trained YOLO model weights for droplet detection.

## 📦 File Naming

- Files are named as `best_X.pt`, where `X` corresponds to the training run or experiment number.
- Use descriptive names if you add weights for specific tasks (e.g., `static_image_best.pt`).

## ➕ Adding New Weights

1. Save new model weights in this folder.
2. Use a clear, descriptive name.
3. Update the relevant configuration or script (such as `PARAMETERS.py` in `static-image-v1` or `webcam-v1`) to reference the new file.

## 🔄 Usage

- Detection scripts load the model weights specified in their `PARAMETERS.py` file.
- To change the active model, update the `WEIGHT` variable in the relevant parameters file.

## 📊 Model Provenance

| File Name   | Date       | Training Data | Config File    | mAP/F1 | Notes                       |
|-------------|------------|---------------|---------------|--------|-----------------------------|
| best_7.pt   | YYYY-MM-DD | dataset_v7    | train_1.yaml  | 0.92   | Best for static images      |
| best_10.pt  | YYYY-MM-DD | dataset_v10   | train_2.yaml  | 0.95   | Trained with augmentations  |
| ...         | ...        | ...           | ...           | ...    | ...                         |

> **Tip:** Update this table when you add new weights to keep track of performance and provenance.

## 📝 Notes

- Large weight files are typically not versioned in git. Use external storage or DVC if needed.
- Remove obsolete weights periodically to keep the folder organized.
- For questions or contributions, see the main project README for contact info and guidelines.

---

*This README helps you organize and track model weights for reproducibility and future development.*
