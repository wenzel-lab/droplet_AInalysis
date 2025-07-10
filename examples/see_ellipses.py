decision = input("1. Make only the image in PARAMETERS.py\n2. Make ALL images in real_imgs\n-> ")

from ultralytics import YOLO as Yolo
from os import path, listdir, remove, mkdir
import cv2
# All parameters for detection and analysis are loaded from PARAMETERS.py
from PARAMETERS import TEST_IMAGE, IMGSZ, CONFIDENCE, MAX_DETECT, PIXEL_RATIO, UNIT, WEIGHT
from data_management.get_boxes import get_boxes
import imageio.v2 as imageio


def numerically(string : str):
    number = ""
    for character in string:
        if character in "0123456789":
            number += character
    return int(number)

if not path.exists(path.join("imgs", "results")):
    mkdir(path.join("imgs", "results"))

if decision == "2":
    # Batch mode: process all images with all weights
    image_paths = [path.join("imgs", "real_imgs", f) 
                   for f in listdir(path.join("imgs", "real_imgs")) 
                   if path.isfile(path.join("imgs", "real_imgs", f))]
else:
    # Single-image mode: process TEST_IMAGE with WEIGHT from PARAMETERS.py
    image_paths = [path.join("imgs", "real_imgs", TEST_IMAGE)]

for image_path in image_paths:
    image_name = image_path.split("\\")[-1]
    output_dir = path.join("imgs", "results", image_name.split(".")[0])
    if not path.exists(output_dir):
        mkdir(output_dir)

    weights_dir = "weights"
    weights_paths = [path.join(weights_dir, f) for f in listdir(weights_dir) if path.isfile(path.join(weights_dir, f))]
    weights_paths = sorted(weights_paths, key=numerically)

    # --- GIF: always show all weights for each image ---
    final_images_paths = []
    for weight in weights_paths:
        model = Yolo(weight)
        results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT, verbose=False)
        img = cv2.imread(image_path)
        output_path = get_boxes(results, img, image_name, weight)
        final_images_paths.append(output_path)

    frames = [imageio.imread(image_file) for image_file in final_images_paths]
    frames.append(frames[-1])
    frames.append(frames[-1])
    imageio.mimsave(path.join(output_dir, "history.gif"), frames, duration=len(weights_paths)*90, loop=0)

    # --- Single-image mode: also save annotated image for WEIGHT only ---
    if decision != "2":
        model = Yolo(path.join("weights", WEIGHT))
        results = model.predict(image_path, imgsz=IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT, verbose=False)
        img = cv2.imread(image_path)
        output_path = get_boxes(results, img, image_name, WEIGHT)
        cv2.imwrite(path.join(output_dir, f"{image_name.split('.')[0]}_{WEIGHT.replace('.pt','')}.jpg"), img)

    for image_path in final_images_paths[:-1]:
        remove(image_path)
