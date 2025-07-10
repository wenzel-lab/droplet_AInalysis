from ultralytics import YOLO as Yolo
from os import path, listdir, remove, mkdir
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plot
from PARAMETERS import TEST_IMAGE, IMGSZ as RAW_IMGSZ, CONFIDENCE, MAX_DETECT, PIXEL_RATIO, UNIT, WEIGHT
from data_management.get_boxes import get_boxes
from data_management.get_distributions import get_dimensions
from math import sqrt, pi
import numpy as np
from scipy import stats
from statistics import mean, stdev, mode, StatisticsError
import csv
import torch
import shutil
from glob import glob

def round_imgsz(imgsz, stride=32):
    return int(np.ceil(imgsz / stride) * stride)

IMGSZ = round_imgsz(RAW_IMGSZ)

def numerically(string: str):
    return int(''.join([c for c in string if c.isdigit()]))

def score_result(result):
    """
    Returns the sum of confidence scores for valid droplet detections only.
    Applies the same filtering used in get_boxes() and get_dimensions():
    - Removes boxes touching image edges
    """
    # Get image size (height and width)
    img_height, img_width = result[0].orig_shape

    # Get bounding boxes from the YOLO result
    array = result[0].boxes

    # If no boxes detected, return score = 0
    if array is None or len(array) == 0:
        return 0

    # Apply same filter as get_boxes(): exclude edge boxes
    mask = (array.xyxy[:, 0] > 1) & (array.xyxy[:, 1] > 1) & \
           (array.xyxy[:, 2] < img_width - 1) & (array.xyxy[:, 3] < img_height - 1)

    filtered = array[mask]

    # Return sum of confidence scores for filtered boxes
    if filtered is None or len(filtered) == 0:
        return 0

    return float(filtered.conf.sum())

def compute_volume(diameter_um):
    r = diameter_um / 2
    return (4/3) * pi * (r ** 3) * 1e-3  # µm³ to pL

def draw_detections(image, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        cv2.ellipse(image, center, axes, 0, 0, 360, (0, 255, 0), 1)
    return image

def overlay_info_on_image(image, weight_name, num_droplets, mean_conf, boxes=None):
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    image = image.copy()
    if boxes is not None:
        image = draw_detections(image, boxes)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"Model: {weight_name}", (10, 25), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"Droplets: {num_droplets}", (10, 50), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"Mean confidence: {mean_conf:.2f}", (10, 75), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return image

def count_valid_droplets(result):
    """Count boxes inside image bounds (same filter as get_boxes/get_dimensions)"""
    img_height, img_width = result[0].orig_shape
    boxes = result[0].boxes
    if boxes is None:
        return 0
    mask = (boxes.xyxy[:, 0] > 1) & (boxes.xyxy[:, 1] > 1) & \
           (boxes.xyxy[:, 2] < img_width - 1) & (boxes.xyxy[:, 3] < img_height - 1)
    return int(mask.sum().item())

def clean_output_directory(output_dir):
    if path.exists(output_dir):
        shutil.rmtree(output_dir)
    mkdir(output_dir)
    for name in ["best_prediction.jpg", "history.gif", "droplet_measurements.csv", "droplet_statistics.png", "model_performance.png"]:
        p = path.join(output_dir, name)
        if path.exists(p): remove(p)
    for f in glob(path.join(output_dir, "latest_prediction_*.jpg")):
        remove(f)

def plot_model_performance(stats, output_dir):
    weights = [s['weight'] for s in stats]
    droplet_counts = [s['droplets'] for s in stats]
    mean_confidences = [s['mean_conf'] for s in stats]

    fig, ax1 = plot.subplots(figsize=(10, 5))
    ax1.set_xlabel('Model weight')
    ax1.set_ylabel('Droplet count', color='tab:blue')
    ax1.plot(weights, droplet_counts, marker='o', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Mean confidence', color='tab:red')
    ax2.plot(weights, mean_confidences, marker='s', linestyle='--', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plot.title('Model Performance: Droplets vs Confidence')
    fig.tight_layout()
    plot.savefig(path.join(output_dir, "model_performance.png"))
    plot.close()

def show_graphics(image_data, output_dir, results):
    
    # Filtered boxes: same logic used across whole pipeline
    boxes = filter_boxes(results)

    # Calculate raw diameters in pixels
    diameters_px = [(((b.xyxy[0][2] - b.xyxy[0][0]) + (b.xyxy[0][3] - b.xyxy[0][1])) / 2).item() for b in boxes]
    diameters_um = [d * PIXEL_RATIO for d in diameters_px]
    volumes_pl = [compute_volume(d) for d in diameters_um]

    # Summary statistics
    d_mean = mean(diameters_um) if diameters_um else 0
    d_std = stdev(diameters_um) if len(diameters_um) > 1 else 0
    v_mean = mean(volumes_pl) if volumes_pl else 0
    v_std = stdev(volumes_pl) if len(volumes_pl) > 1 else 0

    try:
        d_mode = mode(diameters_um)
    except StatisticsError:
        d_mode = round(d_mean, 2)

    try:
        v_mode = mode(volumes_pl)
    except StatisticsError:
        v_mode = round(v_mean, 2)

    n = len(diameters_um)
    unit = image_data.unit

    # Export CSV
    with open(path.join(output_dir, "droplet_measurements.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", f"Diameter ({unit})", "Volume (pL)"])
        for i, (d, v) in enumerate(zip(diameters_um, volumes_pl), 1):
            writer.writerow([i, round(d, 2), round(v, 2)])
        writer.writerow([])
        writer.writerow(["SUMMARY"])
        writer.writerow(["Total droplets", n])
        writer.writerow([f"Mean diameter ({unit})", round(d_mean, 2)])
        writer.writerow([f"Std diameter ({unit})", round(d_std, 2)])
        writer.writerow(["CV diameter", round(d_std / d_mean, 4) if d_mean else "NA"])
        writer.writerow(["Mean volume (pL)", round(v_mean, 2)])
        writer.writerow(["Std volume (pL)", round(v_std, 2)])

    # Plot histograms
    fig, axs = plot.subplots(1, 2, figsize=(12, 5), num=f"{n} droplets detected")
    fig.suptitle(f"{n} droplets detected", fontsize=14)

    # Diameter histogram
    x_diam = np.linspace(min(diameters_um), max(diameters_um), 100)
    y_diam = stats.norm.pdf(x_diam, d_mean, d_std)
    axs[0].hist(diameters_um, bins=15, color='skyblue', alpha=0.6, label='Data')
    axs[0].plot(x_diam, y_diam * n * (x_diam[1] - x_diam[0]), color='darkblue', lw=2, label='Normal Fit')
    axs[0].set_title(f"DIAMETER\nμ = {round(d_mean,2)} {unit}, σ = {round(d_std,2)} {unit}, CV = {round(d_std/d_mean,3) if d_mean else 'NA'}, mode = {round(d_mode,2)} {unit}")
    axs[0].set_xlabel(f"Diameter ({unit})")
    axs[0].set_ylabel("Count")
    axs[0].legend()

    # Volume histogram
    x_vol = np.linspace(min(volumes_pl), max(volumes_pl), 100)
    y_vol = stats.norm.pdf(x_vol, v_mean, v_std)
    axs[1].hist(volumes_pl, bins=15, color='salmon', alpha=0.6, label='Data')
    axs[1].plot(x_vol, y_vol * n * (x_vol[1] - x_vol[0]), color='darkred', lw=2, label='Normal Fit')
    axs[1].set_title(f"VOLUME\nμ = {round(v_mean,2)} pL, σ = {round(v_std,2)} pL , mode = {round(v_mode,2)} pL")
    axs[1].set_xlabel("Volume (pL)")
    axs[1].set_ylabel("Count")
    axs[1].legend()
    
    plot.tight_layout()
    plot.savefig(path.join(output_dir, "droplet_statistics.png"))
    plot.close()

def filter_boxes(results):
    """Apply consistent filtering to YOLO box predictions."""
    img_h, img_w = results[0].orig_shape
    boxes = results[0].boxes
    if boxes is None:
        return []
    mask = (boxes.xyxy[:, 0] > 1) & (boxes.xyxy[:, 1] > 1) & \
           (boxes.xyxy[:, 2] < img_w - 1) & (boxes.xyxy[:, 3] < img_h - 1)
    return boxes[mask]

# ::::::::: Main Script ::::::::::
print("Select image mode:")
print("1. Use TEST_IMAGE from PARAMETERS.py")
print("2. Analyze ALL images in real_imgs folder")
image_mode = input("-> ")

print("Select model mode:")
print("1. Use WEIGHT from PARAMETERS.py")
print("2. Test ALL weights and select best by score")
model_mode = input("-> ")

# Ensure results directory exists
if not path.exists(path.join("imgs", "results")):
    mkdir(path.join("imgs", "results"))

# Determine image(s) to analyze
image_paths = [path.join("imgs/real_imgs", f) for f in listdir("imgs/real_imgs")] if image_mode == "2" \
               else [path.join("imgs/real_imgs", TEST_IMAGE)]

# Determine model(s) to analyze
if model_mode == "1":
    weights_paths = [path.join("weights", WEIGHT)]
else:
    weights_dir = "weights"
    weights_paths = sorted([path.join(weights_dir, f)
                            for f in listdir(weights_dir)
                            if path.isfile(path.join(weights_dir, f))],
                           key=numerically)

# Process each image
for image_path in image_paths:
    image_name = path.basename(image_path)
    output_dir = path.join("imgs", "results", image_name.split(".")[0])
    clean_output_directory(output_dir)

    gif_frames = []
    performance_stats = []
    best_score = -1
    best_weight_name = ""

    for weight_path in weights_paths:
        weight_name = path.basename(weight_path)
        model = Yolo(weight_path)
        results = model.predict(image_path, imgsz=IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT, verbose=False)

        filtered_boxes = filter_boxes(results)
        num_droplets = len(filtered_boxes)
        mean_conf = float(torch.mean(filtered_boxes.conf)) if num_droplets else 0.0

        performance_stats.append({
            'weight': weight_name,
            'droplets': num_droplets,
            'mean_conf': mean_conf
        })

        img = cv2.imread(image_path)
        get_boxes(results, img.copy(), image_name, weight_path)
        overlayed = overlay_info_on_image(img.copy(), weight_name, num_droplets, mean_conf, filtered_boxes)
        gif_frames.append(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))

        if model_mode == "2":  # Only score models if in best-score mode
            score = score_result(results)
            if score > best_score:
                best_score = score
                best_weight_name = weight_name

    # Save GIF and plot performance
    gif_frames.extend([gif_frames[-1]] * 2)
    imageio.mimsave(path.join(output_dir, "history.gif"), gif_frames, duration=len(gif_frames)*90, loop=0)
    plot_model_performance(performance_stats, output_dir)

    # Decide which weight to use for final analysis
    final_weight = WEIGHT if model_mode == "1" else best_weight_name
    print(f"Best prediction for {image_name}: {final_weight}")

    # Predict again with selected model
    best_model = Yolo(path.join("weights", final_weight))
    results = best_model.predict(image_path, imgsz=IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT, verbose=False)

    filtered_boxes = filter_boxes(results)
    num_droplets = len(filtered_boxes)
    mean_conf = float(torch.mean(filtered_boxes.conf)) if num_droplets else 0.0

    img_best = cv2.imread(image_path)
    get_boxes(results, img_best, image_name, final_weight)
    img_annotated = overlay_info_on_image(img_best.copy(), final_weight, num_droplets, mean_conf, filtered_boxes)
    cv2.imwrite(path.join(output_dir, "best_prediction.jpg"), img_annotated)

    image_data = get_dimensions(results, PIXEL_RATIO, UNIT)
    print(f"{image_name} → {image_data.n_droplets[0]} droplets detected.")
    show_graphics(image_data, output_dir, results)