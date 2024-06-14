from scipy import stats
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.widgets import Button
from math import pi, sqrt
from data_management.get_boxes import get_boxes
from PARAMETERS import WEIGHT


def group_in_intervals(bars, interval):
    new_list = []
    for messure, q in bars:
        new_messure = round(messure/interval)*interval
        if not len(new_list) or new_messure > new_list[-1][0]:
            new_list.append([new_messure, q])
        else:
            new_list[-1][1] += q
    return new_list

def exit_program(_, events):
    events["exit"].set()

def forget(_, events):
    events["forget"].set()

def pause(_, events):
    if not events["pause"].is_set():
        events["pause"].set()
    else:
        events["pause"].clear()

constant = 1/sqrt(2*pi)
def plot_bar_with_normal(ax, bars, mean, std, title, unit, n, i, pixel_ratio, interval = 1):
    ax.clear()
    categories = []
    quantities = []
    if title == "Area":
        pixel_ratio *= pixel_ratio
    for key, value in bars:
        categories.append(key * pixel_ratio)
        quantities.append(value)

    ax.bar(categories, quantities, width = interval * pixel_ratio * 0.8, color='skyblue')
    if len(categories):
        ax.set_xlim(0, categories[-1]*1.1)
    else:
        ax.set_xlim(0, 10)

    x = np.linspace(max(mean - 3.5*std, 0), mean + 3.5*std, 100)
    y = stats.norm.pdf(x, mean, std)
    height = n * interval * pixel_ratio
    ax.plot(x, y*height, color='darkblue')
    if len(categories) and std!=0:
        ax.set_ylim(0, max(quantities + [height*(constant)/std])*1.1)
    else:
        ax.set_ylim(0, 1)
    squared = "" if title != "Area" else "²"

    ax.set_xlabel("|  μ = " + str(mean) + " " + unit + squared +"  |  |  σ = " + str(std) + " " + unit + squared + "  |")
    ax.set_title(title)
    if title == "Width":
        ax.set_ylabel("Quantity")
        ax.text(0.02, 0.98, "Droplets per img: " + str(round(n/i,2)), 
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    if title == "Area":
        ax.text(0.02, 0.98, "Interval size: " + str(round(interval*pixel_ratio,2)) + " " + unit + "²", 
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

def show_graphics(events, main_queue, pixel_ratio):
    plot.ion()
    fig, axs = plot.subplots(1, 3, figsize=(12, 5), num="Droplet AInalysis")

    ax_e = plot.axes([0.005, 0.85, 0.05, 0.07])
    btn_e = Button(ax_e, 'Exit')
    btn_e.on_clicked(lambda _: exit_program(_, events))

    ax_f = plot.axes([0.005, 0.75, 0.05, 0.07])
    btn_f = Button(ax_f, 'Forget')
    btn_f.on_clicked(lambda _: forget(_, events))

    ax_p = plot.axes([0.005, 0.65, 0.05, 0.07])
    btn_p = Button(ax_p, 'Pause')
    btn_p.on_clicked(lambda _: pause(_, events))

    while not events["exit"].is_set():
        updated = False
        if not main_queue.empty():
            updated = True

            image_data = main_queue.get()
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
            i = image_data.images_added
            fig.suptitle(f'{i} images considered')

            plot_bar_with_normal(axs[0], width_bars, width_mean, width_std, "Width", unit, n, i, pixel_ratio)
            plot_bar_with_normal(axs[1], height_bars, height_mean, height_std, 'Height', unit, n, i, pixel_ratio)
            plot_bar_with_normal(axs[2], area_bars, area_mean, area_std, 'Area', unit, n, i, pixel_ratio, area_interval)
            # get_boxes(results, img, "a.jpg", WEIGHT, False)

        if updated:
            plot.pause(0.25)
