from PIL import Image
from os.path import join
from math import pi, sqrt
import numpy as np
try:
    from ImageData import ImageData
    from data_tools import sort_and_group
except Exception:
    from data_management.ImageData import ImageData
    from data_tools import sort_and_group


def get_dimentions(results, image_path, pixel_ratio, unit, omit_border_droplets):
    img_width, img_height = Image.open(image_path).size
    ellipse_constant = pi * 0.25

    maximum = 0
    minimum = img_width*img_height
    coordinates = []
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        if (x1 > 1 and y1 > 1 and x2 < img_width-1 and y2 < img_height-1) or not omit_border_droplets:
            coordinates.append((x1, y1, x2, y2))
            area = (x2 - x1) * (y2-y1)
            if area > maximum:
                maximum = area
            if area < minimum:
                minimum = area
    if maximum:
        area_interval = round(sqrt((maximum-minimum) * ellipse_constant))
    else:
        area_interval = 1

    widths = []
    heights = []
    areas = []
    width_sums = [0, 0]
    height_sums = [0, 0]
    area_sums = [0, 0]
    droplets = 0
    for box in coordinates:
        x1, y1, x2, y2 = box
        width = float(x2 - x1)
        height = float(y2 - y1)
        area = float(height * width * ellipse_constant)

        width_sums[0] += width
        width_sums[1] += width**2

        height_sums[0] += height
        height_sums[1] += height**2

        area_sums[0] += area
        area_sums[1] += area**2

        widths.append(round(width))
        heights.append(round(height))

        interval = round(area / area_interval) * area_interval
        areas.append(interval)
        droplets += 1

    width_bars = sort_and_group(widths)
    height_bars = sort_and_group(heights)
    area_bars = sort_and_group(areas)

    return ImageData(droplets, width_sums, height_sums, area_sums, 
                           width_bars, height_bars, area_bars, area_interval,
                           pixel_ratio, unit)

if __name__ == "__main__":
    from ultralytics import YOLO as Yolo
    from PARAMETERS import PIXEL_RATIO, UNIT, IMGSZ, CONFIDENCE, TEST_IMAGE, TEST_WEIGHT, MAX_DETECT, OMIT_BORDER_DROPLETS

    image_path = join("testing_imgs",TEST_IMAGE)
    model = Yolo(join("weights", TEST_WEIGHT))
    results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)
    image_info = get_dimentions(results, image_path, PIXEL_RATIO, UNIT, OMIT_BORDER_DROPLETS)
    print(image_info)