from ultralytics import YOLO as Yolo
from math import pi, sqrt
from os.path import join
from data_management.get_distributions import get_dimensions
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

    diameter_bars = image_data.diameter_bars[0]
    diameter_mean = image_data._diameter_distribution[2]
    diameter_std = image_data._diameter_distribution[3]

    unit = image_data.unit
    n = image_data.n_droplets[0]

    fig, axs = plot.subplots(1, 1, figsize=(15, 5), num=str(n) + " Droplets Detected")
    plot_bar_with_normal(axs, diameter_bars, diameter_mean, diameter_std, 'DIAMETER', unit, n, PIXEL_RATIO) 
    
    plot.show()

image_path = join("imgs", "real_imgs",TEST_IMAGE)
model = Yolo(join("weights", WEIGHT))

results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)
image_data = get_dimensions(results, PIXEL_RATIO, UNIT)
print(image_data)
show_graphics(image_data)
