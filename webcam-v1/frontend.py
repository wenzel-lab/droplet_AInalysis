import numpy as np
import matplotlib.pyplot as plot
from matplotlib.widgets import Button


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

def plot_bar(ax, bars, mean, std, title, unit, n, i, pixel_ratio, interval=1):
    """Render histogram + normal fit matching the style of static-image-v1/video-v1."""
    ax.clear()

    # Choose colour palette consistent with other pipelines
    bar_color = 'skyblue' if title == "DIAMETER" else 'salmon'
    line_color = 'r'

    # Adjust pixel_ratio for volume conversion (µm³→pL)
    display_ratio = pixel_ratio if title == "DIAMETER" else 1
    categories, quantities = [], []
    max_quantity, mode_val = 0, 0
    for size, qty in bars:
        size_disp = size * display_ratio
        categories.append(size_disp)
        quantities.append(qty)
        if qty > max_quantity:
            max_quantity = qty
            mode_val = size_disp

    # Histogram
    ax.bar(categories, quantities, width=interval * display_ratio * 0.8, color=bar_color, edgecolor='black', alpha=0.7)


    # Text box with stats (top-right)
    cv = (std / mean) if mean else 0
    stats_text = (f'Droplets: {n}\n'
                  f'Frames: {i}\n'
                  f'Mean: {mean:.2f} {unit}\n'
                  f'Std Dev: {std:.2f} {unit}\n')
    if title == "DIAMETER":
        stats_text += f'CV: {cv:.2f}\n'
    stats_text += f'Mode: {mode_val:.2f} {unit}'

    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(f'{title.capitalize()} Distribution')
    ax.set_xlabel(f'{title.capitalize()} ({unit})')
    ax.set_ylabel('Count')
    # Make x-axis labels integers when dealing with volume
    if title == 'VOLUME':
        import matplotlib.ticker as mtick
        ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.2)

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
    last_n = 0
    last_i = 0
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
            last_n = n
            last_i = i

            plot_bar(axs[1], diameter_bars, diameter_mean, diameter_std, "DIAMETER", image_data.diameter_unit, n, i, pixel_ratio)
            plot_bar(axs[2], volume_bars, volume_mean, volume_std, 'VOLUME', image_data.volume_unit, n, i, pixel_ratio, volume_interval)
        # Update live view title with latest counts
        axs[0].set_xlabel(f'Droplets detected: {last_n}    |    Frames processed: {last_i}')
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

            plot_bar(axs[0], diameter_bars, diameter_mean, diameter_std, "DIAMETER", image_data.diameter_unit, n, i, pixel_ratio)
            plot_bar(axs[1], volume_bars, volume_mean, volume_std, 'VOLUME', image_data.volume_unit, n, i, pixel_ratio, volume_interval)

        if updated:
            plot.pause(0.2)
