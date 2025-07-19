import cv2
import os
import sys
import logging
from PARAMETERS import IMGSZ, CONFIDENCE, PIXEL_RATIO, WEIGHT, VIDEO_TO_ANALYZE
from data_management.get_boxes import get_boxes
from data_management.get_distributions import get_dimensions
import torch
from ultralytics import YOLO as Yolo
import numpy as np
import csv

# Suppress most logging from ultralytics, torch, and root logger
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
logging.getLogger('torch').setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)

SAMPLES_DIR = os.path.join('videos', 'samples')
RESULTS_DIR = os.path.join('videos', 'results')


def get_video_files(samples_dir):
    return [os.path.join(samples_dir, f) for f in os.listdir(samples_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.h264'))]

import contextlib

def process_video(video_path, model, max_frames=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    result_root = os.path.join(RESULTS_DIR, video_name)
    predictions_dir = os.path.join(result_root, 'predictions')
    statistics_dir = os.path.join(result_root, 'statistics')
    annotated_frames_dir = os.path.join(predictions_dir, 'annotated_frames')
    # Ensure folder hierarchy exists
    os.makedirs(annotated_frames_dir, exist_ok=True)
    os.makedirs(statistics_dir, exist_ok=True)
    # Alias for existing code that references 'output_dir' for statistics
    output_dir = statistics_dir

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = os.path.join(predictions_dir, f"annotated_{video_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    csv_path = os.path.join(statistics_dir, 'droplet_measurements.csv')
    droplet_counts = []
    measurement_idx = 1
    frame_idx = 0
    print(f"Processing video for {video_name}...")

    # For summary stats
    all_diameters = []
    all_volumes = []
    per_droplet_rows = []

    from tqdm import tqdm
    # Try to get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0 or total_frames > 1000000:  # unreliable or huge
        total_frames = None
    if max_frames is not None and total_frames is not None:
        total = min(max_frames, total_frames)
    elif max_frames is not None:
        total = max_frames
    else:
        total = total_frames

    with tqdm(total=total, desc=f"Analyzing frames", unit="frame") as pbar:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            img = frame  # No resizing, use original resolution
            # Suppress YOLO/torch output (Python-level)
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    results = model(img)
            boxes_img = img.copy()
            get_boxes(results, boxes_img, f"frame_{frame_idx}", WEIGHT)
            # Save per-frame annotated image
            frame_img_path = os.path.join(annotated_frames_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(frame_img_path, boxes_img)
            # Collect droplet measurements
            if results and results[0].boxes is not None:
                array = results[0].boxes
                widths_vector = (array.xyxy[:, 2] - array.xyxy[:, 0])
                heights_vector = (array.xyxy[:, 3] - array.xyxy[:, 1])
                diameters_vector = ((widths_vector + heights_vector)/2) * PIXEL_RATIO
                sphere_constant = np.pi / 6
                volumes_vector = (diameters_vector ** 3) * sphere_constant * 1e-3  # um^3 to pL
                for d, v in zip(diameters_vector, volumes_vector):
                    per_droplet_rows.append([measurement_idx, round(float(d), 2), round(float(v), 2)])
                    all_diameters.append(float(d))
                    all_volumes.append(float(v))
                    measurement_idx += 1
                droplet_counts.append(len(array))
            else:
                droplet_counts.append(0)
            # Write frame to output video
            out.write(boxes_img)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()

    # Compute summary stats
    total_droplets = len(all_diameters)
    mean_diameter = np.mean(all_diameters) if all_diameters else 0
    std_diameter = np.std(all_diameters, ddof=1) if len(all_diameters) > 1 else 0
    cv_diameter = (std_diameter / mean_diameter) if mean_diameter else 0
    mean_volume = np.mean(all_volumes) if all_volumes else 0
    std_volume = np.std(all_volumes, ddof=1) if len(all_volumes) > 1 else 0
    
    # Calculate mode using histogram bins
    from scipy import stats
    mode_diameter = 0
    mode_volume = 0
    if all_diameters:
        mode_result = stats.mode(np.round(all_diameters, 1), keepdims=True)
        mode_diameter = mode_result.mode[0] if len(mode_result.mode) > 0 else 0
    if all_volumes:
        mode_result = stats.mode(np.round(all_volumes, 1), keepdims=True)
        mode_volume = mode_result.mode[0] if len(mode_result.mode) > 0 else 0

    # Write summary and data to CSV
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['SUMMARY'])
        csv_writer.writerow(['Total droplets', total_droplets])
        csv_writer.writerow(['Mean diameter (um)', round(mean_diameter, 2)])
        csv_writer.writerow(['Std diameter (um)', round(std_diameter, 2)])
        csv_writer.writerow(['Mode diameter (um)', round(mode_diameter, 2)])
        csv_writer.writerow(['CV diameter', round(cv_diameter, 4)])
        csv_writer.writerow(['Mean volume (pL)', round(mean_volume, 2)])
        csv_writer.writerow(['Std volume (pL)', round(std_volume, 2)])
        csv_writer.writerow(['Mode volume (pL)', round(mode_volume, 2)])
        csv_writer.writerow([])
        csv_writer.writerow(['Index', 'Diameter (um)', 'Volume (pL)'])
        csv_writer.writerows(per_droplet_rows)

    # Generate and save statistics plots (diameter and volume histograms)
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    if all_diameters:
        # Statistics text box for all plots
        stats_text = f'Droplets detected: {total_droplets}\n'
        stats_text += f'Mean: {mean_diameter:.2f} μm\n'
        stats_text += f'Std Dev: {std_diameter:.2f} μm\n'
        stats_text += f'CV: {cv_diameter:.2f}\n'
        stats_text += f'Mode: {mode_diameter:.2f} μm'
        
        # 1. Mixed plot: Histogram with normal fit overlay
        plt.figure(figsize=(8,6))
        n, bins, patches = plt.hist(all_diameters, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        
        # Add normal fit curve scaled to match histogram counts
        x = np.linspace(min(all_diameters), max(all_diameters), 100)
        bin_width = (max(all_diameters) - min(all_diameters)) / 30
        normal_fit = norm.pdf(x, mean_diameter, std_diameter) * len(all_diameters) * bin_width
        plt.plot(x, normal_fit, 'r-', linewidth=2, label='Normal fit')
        
        plt.title('Droplet Diameter Distribution (Mixed)')
        plt.xlabel('Diameter (μm)')
        plt.ylabel('Count')
        plt.legend()
        
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'diameter_mixed.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Histogram only
        plt.figure(figsize=(8,6))
        plt.hist(all_diameters, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        
        plt.title('Droplet Diameter Distribution (Histogram)')
        plt.xlabel('Diameter (μm)')
        plt.ylabel('Count')
        
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'diameter_hist.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Normal fit only
        plt.figure(figsize=(8,6))
        x = np.linspace(min(all_diameters), max(all_diameters), 100)
        normal_fit = norm.pdf(x, mean_diameter, std_diameter)
        plt.plot(x, normal_fit, 'r-', linewidth=3)
        
        plt.title('Droplet Diameter Distribution (Normal Fit)')
        plt.xlabel('Diameter (μm)')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'diameter_normal.png'), dpi=300, bbox_inches='tight')
        plt.close()
    if all_volumes:
        # Statistics text box for all plots
        stats_text = f'Droplets detected: {total_droplets}\n'
        stats_text += f'Mean: {mean_volume:.2f} pL\n'
        stats_text += f'Std Dev: {std_volume:.2f} pL\n'
        stats_text += f'Mode: {mode_volume:.2f} pL'
        
        # 1. Mixed plot: Histogram with normal fit overlay
        plt.figure(figsize=(8,6))
        n, bins, patches = plt.hist(all_volumes, bins=30, color='salmon', edgecolor='black', alpha=0.7)
        
        # Add normal fit curve scaled to match histogram counts
        x = np.linspace(min(all_volumes), max(all_volumes), 100)
        bin_width = (max(all_volumes) - min(all_volumes)) / 30
        normal_fit = norm.pdf(x, mean_volume, std_volume) * len(all_volumes) * bin_width
        plt.plot(x, normal_fit, 'r-', linewidth=2, label='Normal fit')
        
        plt.title('Droplet Volume Distribution (Mixed)')
        plt.xlabel('Volume (pL)')
        plt.ylabel('Count')
        plt.legend()
        
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'volume_mixed.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Histogram only
        plt.figure(figsize=(8,6))
        plt.hist(all_volumes, bins=30, color='salmon', edgecolor='black', alpha=0.7)
        
        plt.title('Droplet Volume Distribution (Histogram)')
        plt.xlabel('Volume (pL)')
        plt.ylabel('Count')
        
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'volume_hist.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Normal fit only
        plt.figure(figsize=(8,6))
        x = np.linspace(min(all_volumes), max(all_volumes), 100)
        normal_fit = norm.pdf(x, mean_volume, std_volume)
        plt.plot(x, normal_fit, 'r-', linewidth=3)
        
        plt.title('Droplet Volume Distribution (Normal Fit)')
        plt.xlabel('Volume (pL)')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'volume_normal.png'), dpi=300, bbox_inches='tight')
        plt.close()
    # Save summary statistics
    stats_path = os.path.join(statistics_dir, 'video_analysis_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Total frames processed: {frame_idx}\n")
        f.write(f"Droplet counts per frame: {droplet_counts}\n")
        if droplet_counts:
            f.write(f"Mean droplets per frame: {np.mean(droplet_counts):.2f}\n")
        else:
            f.write("Warning: No frames were processed. Check if the video format is supported and not corrupted.\n")
    print(f"Results for {video_name} saved to {result_root}")
    print(f"Annotated video: {out_video_path}")
    print(f"CSV: {csv_path}")
    print(f"Stats: {stats_path}")
    if droplet_counts:
        print(f"Mean droplets per frame: {np.mean(droplet_counts):.2f}")
    else:
        print("Warning: No frames were processed. Check if the video format is supported and not corrupted.")

def main():
    print("DROPLET VIDEO ANALYSIS - BATCH MODE")
    if not os.path.exists(SAMPLES_DIR):
        print(f"Samples directory not found: {SAMPLES_DIR}")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Loading YOLO model...")
    weight_path = os.path.join('weights', WEIGHT)
    model = Yolo(weight_path)
    # Determine which videos to process
    if VIDEO_TO_ANALYZE and VIDEO_TO_ANALYZE.strip():
        video_path = os.path.join(SAMPLES_DIR, VIDEO_TO_ANALYZE)
        if not os.path.isfile(video_path):
            print(f"Selected video '{VIDEO_TO_ANALYZE}' not found in {SAMPLES_DIR}.")
            return
        video_files = [video_path]
        print(f"Processing only selected video: {VIDEO_TO_ANALYZE}")
    else:
        video_files = get_video_files(SAMPLES_DIR)
        if not video_files:
            print(f"No video files found in {SAMPLES_DIR}")
            return
        print(f"Processing all {len(video_files)} videos in {SAMPLES_DIR}.")

    print("How many frames to analyze per video? (Enter a number, or press Enter to analyze all frames): ", end="")
    user_input = input().strip()
    max_frames = None
    if user_input:
        try:
            max_frames = int(user_input)
            print(f"Will analyze first {max_frames} frames of each video.")
        except ValueError:
            print("Invalid input. Will analyze all frames.")
    for video_path in video_files:
        process_video(video_path, model, max_frames)
    print("All videos processed.")

if __name__ == "__main__":
    main()
