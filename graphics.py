from time import sleep
from scipy import stats
import numpy as np
from matplotlib.pyplot import plt as plot


def plot_bar_with_normal(ax, bars, mean, std, title, unit, n, interval = 1, intervals_sum = 1):
    categories = []
    quantities = []
    for key, value in bars:
        categories.append(key)
        quantities.append(value)
    height = n * intervals_sum

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

def show_graphics(image_data, events):
    while not events["stop"].is_set():
        if events["data_updated"].is_set():
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
            intervals_sum = image_data.intervals_sum[0]

            unit = image_data.unit
            n = image_data.n_droplets[0]

            fig, axs = plot.subplots(1, 3, figsize=(15, 5), num=str(n) + " Droplets Detected")
            plot_bar_with_normal(axs[0],width_bars, width_mean, width_std, 'Width', unit, n)
            plot_bar_with_normal(axs[1], height_bars, height_mean, height_std, 'Height', unit, n)
            plot_bar_with_normal(axs[2], area_bars, area_mean, area_std, 'Area', unit, n, area_interval, intervals_sum)

            plot.show(block = False)
            
            events["data_updated"].clear()