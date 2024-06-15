from ultralytics import YOLO as Yolo
from math import pi, sqrt
from os.path import join
from data_management.get_distributions import get_dimentions
from PARAMETERS import (TEST_IMAGE, WEIGHT, IMGSZ, 
                        CONFIDENCE, MAX_DETECT, 
                        PIXEL_RATIO, UNIT)
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
    square = ""
    plus_minus = ""
    if title == "AREA":
        pixel_ratio *= pixel_ratio
        square = "²"
        plus_minus = f"± {round(interval*pixel_ratio*0.5, 2)}"
    
    categories = []
    quantities = []
    max_quantity = mode = 0
    for size, quantity in bars:
        categories.append(size * pixel_ratio)
        quantities.append(quantity)
        if quantity > max_quantity:
            max_quantity = quantity
            mode = size
    mode = round(mode * pixel_ratio, 2)

    ax.bar(categories, quantities, width = interval * pixel_ratio * 0.8, color='skyblue')

    if len(categories):
        max_size = categories[-1]
    else:
        max_size = 0
    x = np.linspace(max(mean - 3.5*std, 0), max(mean + 3.5*std, max_size), 100)
    y = stats.norm.pdf(x, mean, std)
    height = n * interval * pixel_ratio
    ax.plot(x, y*height, color='darkblue')

    if len(categories) and std != 0:
        ax.set_ylim(0, max(max_quantity, height*(constant)/std)*1.15)
    else:
        ax.set_ylim(0, 1.15)

    ax.set_title(f"{title}  |  mode = {mode} {plus_minus} {unit}{square}  |")
    ax.set_xlabel(f"|  μ = {mean} {unit}{square}  |  |  σ = {std} {unit}{square}  |")
    if title == "WIDTH":
        ax.set_ylabel("Quantity")
        ax.text(0.02, 0.98, "Droplets: " + str(n), 
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

def show_graphics(image_data):
    width_bars = image_data.width_bars[0]
    width_mean = image_data.width_distribution[2]
    width_std = image_data.width_distribution[3]

    height_bars = image_data.height_bars[0]
    height_mean = image_data.height_distribution[2]
    height_std = image_data.height_distribution[3]

    area_bars = image_data.area_bars[0]
    area_mean = image_data.area_distribution[2]
    area_std = image_data.area_distribution[3]
    area_interval = image_data.area_interval[0]
    area_bars = group_in_intervals(area_bars, area_interval)

    unit = image_data.unit
    n = image_data.n_droplets[0]

    fig, axs = plot.subplots(1, 3, figsize=(15, 5), num=str(n) + " Droplets Detected")
    plot_bar_with_normal(axs[0],width_bars, width_mean, width_std, 'WIDTH', unit, n, PIXEL_RATIO)
    plot_bar_with_normal(axs[1], height_bars, height_mean, height_std, 'HEIGHT', unit, n, PIXEL_RATIO)
    plot_bar_with_normal(axs[2], area_bars, area_mean, area_std, 'AREA', unit, n, PIXEL_RATIO, area_interval)

    plot.show()


image_path = join("imgs", "real_imgs",TEST_IMAGE)
model = Yolo(join("weights", WEIGHT))

results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)
image_data = get_dimentions(results, PIXEL_RATIO, UNIT)
print(image_data)
show_graphics(image_data)