from os.path import join
from get_dimentions import get_dimentions
from get_boxes import get_boxes
from PARAMETERS import (TEST_IMAGE, TEST_WEIGHT, IMGSZ, 
                        CONFIDENCE, MAX_DETECT, PIXEL_RATIO, 
                        UNIT, OMIT_BORDER_DROPLETS, SAVE)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plot


def plot_bar_with_normal(ax, bars, mean, std, title, unit, n, interval = 1):
    categories = []
    quantities = []
    for key, value in bars:
        categories.append(key)
        quantities.append(value)
    height = n * interval

    ax.bar(categories, quantities, width = interval * 0.8, color='skyblue')

    x = np.linspace(max(mean - 3.5*std, 0), mean + 3.5*std, 100)
    y = stats.norm.pdf(x, mean, std)
    ax.plot(x, y*height, color='darkblue')

    squared = ""
    if title == "Area":
        squared = "²"

    ax.set_title(title + " (" + unit + ")")
    ax.set_xlabel("|  μ = " + str(mean) + " " + unit + squared +"  |  |  σ = " + str(std) + " " + unit + squared + "  |")
    if title == "Width":
        ax.set_ylabel("Quantity")

def show_graphics(image_data):
    width_bars = image_data.width_bars[0]
    width_mean = image_data.width_distribution[1]
    width_std = image_data.width_distribution[2]

    height_bars = image_data.height_bars[0]
    height_mean = image_data.height_distribution[1]
    height_std = image_data.height_distribution[2]

    area_bars = image_data.area_bars[0]
    area_mean = image_data.area_distribution[1]
    area_std = image_data.area_distribution[2]
    area_interval = image_data.area_interval[0]

    unit = image_data.unit
    n = image_data.n_droplets[0]

    fig, axs = plot.subplots(1, 3, figsize=(15, 5), num=str(n) + " Droplets Detected")
    plot_bar_with_normal(axs[0],width_bars, width_mean, width_std, 'Width', unit, n)
    plot_bar_with_normal(axs[1], height_bars, height_mean, height_std, 'Height', unit, n)
    plot_bar_with_normal(axs[2], area_bars, area_mean, area_std, 'Area', unit, n, area_interval)

    plot.show()

decision = input("Choose what to do\n1. Get dimentions\n2. Show boxes\n3. Both\nX. Exit\n-> ")

if decision in "123":
    from ultralytics import YOLO as Yolo
    image_path = join("..", "testing_imgs",TEST_IMAGE)
    model = Yolo(join("..", "weights", TEST_WEIGHT))

if decision == "1":
    results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)
    image_info_1 = get_dimentions(results, image_path, PIXEL_RATIO, UNIT, OMIT_BORDER_DROPLETS)
    print(image_info_1)

elif decision == "2":
    results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)
    droplet_cutouts = get_boxes(results, image_path, TEST_IMAGE, TEST_WEIGHT, SAVE, OMIT_BORDER_DROPLETS)

elif decision == "3":
    results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)
    image_info = get_dimentions(results, image_path, PIXEL_RATIO, UNIT, OMIT_BORDER_DROPLETS)
    print(image_info)
    droplet_cutouts = get_boxes(results, image_path, TEST_IMAGE, TEST_WEIGHT, SAVE, OMIT_BORDER_DROPLETS)
