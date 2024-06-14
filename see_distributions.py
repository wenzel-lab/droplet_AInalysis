from ultralytics import YOLO as Yolo
from math import pi, sqrt
from os.path import join
from data_management.get_dimentions import get_dimentions
from PARAMETERS import (TEST_IMAGE, WEIGHT, IMGSZ, 
                        CONFIDENCE, MAX_DETECT, PIXEL_RATIO, 
                        UNIT)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plot


def group_in_intervals(bars, interval):
    new_list = []
    for messure, q in bars:
        new_messure = round(messure/interval)*interval
        if not len(new_list) or new_messure > new_list[-1][0]:
            new_list.append([new_messure, q])
        else:
            new_list[-1][1] += q
    return new_list

constant = 1/sqrt(2*pi)
def plot_bar_with_normal(ax, bars, mean, std, title, unit, n, pixel_ratio = 1, interval = 1):
    categories = []
    quantities = []
    if title == "Area":
        pixel_ratio *= pixel_ratio
    for key, value in bars:
        categories.append(key * pixel_ratio)
        quantities.append(value)

    ax.bar(categories, quantities, width = interval * pixel_ratio * 0.8, color='skyblue')
    x = np.linspace(max(mean - 3.5*std, 0), mean + 3.5*std, 100)
    y = stats.norm.pdf(x, mean, std)
    height = n * interval * pixel_ratio
    ax.plot(x, y*height, color='darkblue')
    if len(categories) and std!=0:
        ax.set_ylim(0, max(quantities + [height*(constant)/std])*1.1)
    else:
        ax.set_ylim(0, 1)
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
    area_bars = group_in_intervals(area_bars, area_interval)

    unit = image_data.unit
    n = image_data.n_droplets[0]

    fig, axs = plot.subplots(1, 3, figsize=(15, 5), num=str(n) + " Droplets Detected")
    plot_bar_with_normal(axs[0],width_bars, width_mean, width_std, 'Width', unit, n, PIXEL_RATIO)
    plot_bar_with_normal(axs[1], height_bars, height_mean, height_std, 'Height', unit, n, PIXEL_RATIO)
    plot_bar_with_normal(axs[2], area_bars, area_mean, area_std, 'Area', unit, n, PIXEL_RATIO, area_interval)

    plot.show()


image_path = join("real_imgs",TEST_IMAGE)
model = Yolo(join("weights", WEIGHT))

results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)
image_data = get_dimentions(results, PIXEL_RATIO, UNIT)
show_graphics(image_data)