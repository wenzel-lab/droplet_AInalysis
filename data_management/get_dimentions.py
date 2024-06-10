from PIL import Image
from os.path import join
from math import pi, sqrt
import torch
try:
    from ImageData import ImageData
    from data_tools import sort_and_group
except Exception:
    from data_management.ImageData import ImageData
    from data_tools import sort_and_group

from time import time


def get_dimentions(results, image_path, pixel_ratio, unit, omit_border_droplets):
    img_width, img_height = Image.open(image_path).size
    ellipse_constant = pi * 0.25

    maximum = 0
    minimum = img_width*img_height

    start = time()
    array = results[0].boxes
    filter = (array.xyxy[:, 0] > 1) & (array.xyxy[:, 1] > 1) & (array.xyxy[:, 2] < img_width - 1) & (array.xyxy[:, 3] < img_height - 1)
    array = array[filter]

    widths_vector = array.xyxy[:, 2] - array.xyxy[:, 0]
    heights_vector = array.xyxy[:, 3] - array.xyxy[:, 1]
    areas_vector = torch.mul(widths_vector, heights_vector) * ellipse_constant
    area_interval = 0
    if len(areas_vector):
        maximum = torch.max(areas_vector)
        minimum = torch.min(areas_vector)
        area_interval = round(sqrt(maximum-minimum))

    width_sums = [widths_vector.sum().item(), (widths_vector**2).sum().item()]
    height_sums = [heights_vector.sum().item(), (heights_vector**2).sum().item()]
    area_sums = [areas_vector.sum().item(), (areas_vector**2).sum().item()]
    widths = torch.round(widths_vector).tolist()
    heights = torch.round(heights_vector).tolist()
    areas = (torch.round(areas_vector/area_interval)*area_interval).tolist()

    width_bars = sort_and_group(widths)
    height_bars = sort_and_group(heights)
    area_bars = sort_and_group(areas)

    print(time() - start, "seconds")

    return ImageData(len(widths_vector), width_sums, height_sums, area_sums, 
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