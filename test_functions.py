from os.path import join
from data_management.get_dimentions import get_dimentions
from data_management.get_boxes import get_boxes
from PARAMETERS import (TEST_IMAGE, WEIGHT, IMGSZ, 
                        CONFIDENCE, MAX_DETECT, PIXEL_RATIO, 
                        UNIT, SAVE)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plot
import cv2


def plot_bar_with_normal(ax, bars, mean, std, title, unit, n, pixel_ratio = 1, interval = 1):
    categories = []
    quantities = []
    if title == "Area":
        pixel_ratio *= pixel_ratio
    for key, value in bars:
        categories.append(key * pixel_ratio)
        quantities.append(value)
    height = n * interval * pixel_ratio

    ax.bar(categories, quantities, width = interval * pixel_ratio * 0.8, color='skyblue')

    x = np.linspace(max(mean - 3.5*std, 0), mean + 3.5*std, 100)
    y = stats.norm.pdf(x, mean, std)
    ax.plot(x, y*height, color='darkblue')

    squared = "" if title != "Area" else "²"

    ax.set_title(title + " (" + unit + squared + ")")
    ax.set_xlabel("|  μ = " + str(mean) + " " + unit + squared +"  |  |  σ = " + str(std) + " " + unit + squared + "  |")
    if title == "Width":
        ax.set_ylabel("Quantity")
        ax.text(0.02, 0.98, "Droplets: " + str(n), 
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    if title == "Area":
        ax.text(0.02, 0.98, "Interval size: " + str(round(interval*pixel_ratio,2)) + " " + unit + "²", 
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

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
    plot_bar_with_normal(axs[0],width_bars, width_mean, width_std, 'Width', unit, n, PIXEL_RATIO)
    plot_bar_with_normal(axs[1], height_bars, height_mean, height_std, 'Height', unit, n, PIXEL_RATIO)
    plot_bar_with_normal(axs[2], area_bars, area_mean, area_std, 'Area', unit, n, PIXEL_RATIO, area_interval)

    plot.show()

decision = input("Choose what to do\n1. Get dimentions\n2. Show boxes\n3. Both\nX. Exit\n-> ")

if decision in "123":
    from ultralytics import YOLO as Yolo
    image_path = join("testing_imgs",TEST_IMAGE)
    model = Yolo(join("weights", WEIGHT))

if decision == "1":
    results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)
    image_info_1 = get_dimentions(results, PIXEL_RATIO, UNIT)
    print(image_info_1)
    show_graphics(image_info_1)

elif decision == "2":
    results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)
    img = cv2.imread(image_path)
    droplet_cutouts = get_boxes(results, img, TEST_IMAGE, WEIGHT, SAVE)

elif decision == "3":
    results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)
    image_info = get_dimentions(results, PIXEL_RATIO, UNIT)
    print(image_info)
    img = cv2.imread(image_path)
    droplet_cutouts = get_boxes(results, img, TEST_IMAGE, WEIGHT, SAVE)
