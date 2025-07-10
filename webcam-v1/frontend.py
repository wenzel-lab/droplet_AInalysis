from scipy import stats
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.widgets import Button
from math import pi, sqrt


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
    plus_minus = ""
    if title == "VOLUME":
        pixel_ratio *= pixel_ratio
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
    # Only plot the normal curve if std is positive and not NaN
    if std > 0 and not np.isnan(std):
        x = np.linspace(max(mean - 3.5*std, 0), max(mean + 3.5*std, max_size), 100)
        y = stats.norm.pdf(x, mean, std)
        height = n * interval * pixel_ratio
        ax.plot(x, y*height, color='darkblue')
        if len(categories):
            ax.set_ylim(0, max(max_quantity, height*(constant)/std)*1.1)
        else:
            ax.set_ylim(0, 1.15)
    else:
        # If std is zero or NaN, skip normal curve and set default y-limits
        ax.set_ylim(0, 1.15)

    ax.set_title(f"{title}  |  mode = {mode} {plus_minus} {unit}  |")
    ax.set_xlabel(f"|  μ = {mean} {unit}  |  |  σ = {std} {unit}  |")
    if title == "DIAMETER":
        ax.set_ylabel("Quantity")
        ax.text(0.019, 0.985, "Droplets/image = " + str(round(n/i)), 
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

def show_combined_ui(model, cap, events, main_queue, pixel_ratio):
    import cv2
    plot.ion()
    fig, axs = plot.subplots(1, 3, figsize=(18, 6), num="Droplet AInalysis: Live + Stats")
    fig.canvas.mpl_connect("close_event", lambda _: exit_program(_, events))

    ax_e = plot.axes([0.005, 0.9, 0.05, 0.07])
    btn_e = Button(ax_e, 'Exit')
    btn_e.on_clicked(lambda _: exit_program(_, events))

    ax_f = plot.axes([0.005, 0.8, 0.05, 0.07])
    btn_f = Button(ax_f, 'Forget')
    btn_f.on_clicked(lambda _: forget(_, events))

    ax_p = plot.axes([0.005, 0.7, 0.05, 0.07])
    btn_p = Button(ax_p, "Pause")
    btn_p.on_clicked(lambda _: pause(_, events))

    frame = None
    while not events["exit"].is_set():
        updated = False
        # --- LIVE VIEW (LEFT) ---
        ret, img = cap.read()
        if ret:
            results = model.predict(img, imgsz=640, conf=0.75, max_det=1000)
            # Draw bounding boxes
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[0].clear()
            axs[0].imshow(frame)
            axs[0].set_title("Live Droplet Detection")
            axs[0].axis('off')
        # --- STATISTICS (CENTER/RIGHT) ---
        if not main_queue.empty():
            updated = True
            image_data = main_queue.get()
            diameter_bars = image_data.diameter_bars[0]
            diameter_mean = image_data.diameter_distribution[2]
            diameter_std = image_data.diameter_distribution[3]

            volume_bars = image_data.volume_bars[0]
            volume_mean = image_data.volume_distribution[2]
            volume_std = image_data.volume_distribution[3]
            volume_interval = image_data.volume_interval[0]
            volume_bars = group_in_intervals(volume_bars, volume_interval)

            n = image_data.n_droplets[0]
            i = image_data.images_added
            fig.suptitle(f'{i} images considered')

            plot_bar_with_normal(axs[1], diameter_bars, diameter_mean, diameter_std, "DIAMETER", image_data.diameter_unit, n, i, pixel_ratio)
            plot_bar_with_normal(axs[2], volume_bars, volume_mean, volume_std, 'VOLUME', image_data.volume_unit, n, i, pixel_ratio, volume_interval)
        if updated or ret:
            plot.pause(0.02)
    plot.ioff()
    plot.close(fig)

def show_graphics(events, main_queue, pixel_ratio):
    plot.ion()
    fig, axs = plot.subplots(1, 2, figsize=(12.5, 5), num="Droplet AInalysis")
    fig.canvas.mpl_connect("close_event", lambda _: exit_program(_, events))

    ax_e = plot.axes([0.005, 0.9, 0.05, 0.07])
    btn_e = Button(ax_e, 'Exit')
    btn_e.on_clicked(lambda _: exit_program(_, events))

    ax_f = plot.axes([0.005, 0.8, 0.05, 0.07])
    btn_f = Button(ax_f, 'Forget')
    btn_f.on_clicked(lambda _: forget(_, events))

    ax_p = plot.axes([0.005, 0.7, 0.05, 0.07])
    btn_p = Button(ax_p, "Pause")
    btn_p.on_clicked(lambda _: pause(_, events))

    while not events["exit"].is_set():
        updated = False
        if not main_queue.empty():
            updated = True

            image_data = main_queue.get()
            diameter_bars = image_data.diameter_bars[0]
            diameter_mean = image_data.diameter_distribution[2]
            diameter_std = image_data.diameter_distribution[3]

            volume_bars = image_data.volume_bars[0]
            volume_mean = image_data.volume_distribution[2]
            volume_std = image_data.volume_distribution[3]
            volume_interval = image_data.volume_interval[0]
            volume_bars = group_in_intervals(volume_bars, volume_interval)

            n = image_data.n_droplets[0]
            i = image_data.images_added
            fig.suptitle(f'{i} images considered')

            plot_bar_with_normal(axs[0], diameter_bars, diameter_mean, diameter_std, "DIAMETER", image_data.diameter_unit, n, i, pixel_ratio)
            plot_bar_with_normal(axs[1], volume_bars, volume_mean, volume_std, 'VOLUME', image_data.volume_unit, n, i, pixel_ratio, volume_interval)

        if updated:
            plot.pause(0.2)
