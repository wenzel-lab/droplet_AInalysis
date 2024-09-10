from math import pi
import torch
from data_management.ImageData import ImageData
from data_management.data_tools import sort_and_group, format

ellipse_constant = pi * 0.25
sphere_constant = pi/6
def get_dimentions(results, pixel_ratio, unit):
    img_height, img_width = results[0].orig_shape

    array = results[0].boxes
    filter = (array.xyxy[:, 0] > 1) & (array.xyxy[:, 1] > 1) & (array.xyxy[:, 2] < img_width - 1) & (array.xyxy[:, 3] < img_height - 1)
    array = array[filter]

    widths_vector = (array.xyxy[:, 2] - array.xyxy[:, 0])
    heights_vector = (array.xyxy[:, 3] - array.xyxy[:, 1])
    diameters_vector = (widths_vector + heights_vector)/2
    volumes_vector = (diameters_vector**3)*sphere_constant

    diameter_sums = [diameters_vector.sum().item() * pixel_ratio, ((diameters_vector*pixel_ratio)**2).sum().item()]
    volume_sums = [volumes_vector.sum().item() * pixel_ratio, ((volumes_vector * pixel_ratio**3)**2).sum().item()]

    diameters = (torch.round(widths_vector)).tolist()
    volumes = (torch.round(volumes_vector)).tolist()

    diameter_bars = sort_and_group(diameters, "DIAMETERS")
    volume_bars = sort_and_group(volumes, "VOLUMES")

    volume_interval = 1
    if len(volumes_vector):
        maximum = torch.max(volumes_vector)
        minimum = torch.min(volumes_vector)
        if diameter_bars[-1][0] != diameter_bars[0][0]:
            volume_interval = int(((maximum - minimum) / (diameter_bars[-1][0] - diameter_bars[0][0])).item())

    return ImageData(format(len(widths_vector)), (img_width, img_height),
                     format(diameter_sums), format(volume_sums), 
                     format(diameter_bars), format(volume_bars), 
                     format(volume_interval), unit)