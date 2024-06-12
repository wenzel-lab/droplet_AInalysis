from math import pi
import torch
from data_management.ImageData import ImageData
from data_management.data_tools import sort_and_group, format

ellipse_constant = pi * 0.25
def get_dimentions(results, pixel_ratio, unit):
    img_height, img_width = results[0].orig_shape

    array = results[0].boxes
    filter = (array.xyxy[:, 0] > 1) & (array.xyxy[:, 1] > 1) & (array.xyxy[:, 2] < img_width - 1) & (array.xyxy[:, 3] < img_height - 1)
    array = array[filter]

    widths_vector = (array.xyxy[:, 2] - array.xyxy[:, 0])
    heights_vector = (array.xyxy[:, 3] - array.xyxy[:, 1])
    areas_vector = torch.mul(widths_vector, heights_vector) * ellipse_constant

    width_sums = [widths_vector.sum().item() *pixel_ratio, ((widths_vector*pixel_ratio)**2).sum().item()]
    height_sums = [heights_vector.sum().item() * pixel_ratio, ((heights_vector* pixel_ratio)**2).sum().item()]
    area_sums = [areas_vector.sum().item()* pixel_ratio**2, ((areas_vector* pixel_ratio**2)**2).sum().item()]
    widths = (torch.round(widths_vector)).tolist()
    heights = (torch.round(heights_vector)).tolist()
    areas = (torch.round(areas_vector)).tolist()

    width_bars = sort_and_group(widths)
    height_bars = sort_and_group(heights)
    area_bars = sort_and_group(areas)

    area_interval = 1
    if len(areas_vector):
        maximum = torch.max(areas_vector)
        minimum = torch.min(areas_vector)
        area_interval = int((2*(maximum - minimum) / (width_bars[-1][0] - width_bars[0][0] + height_bars[-1][0] - height_bars[0][0] + 2)).item())

    return ImageData(format(len(widths_vector)), (img_width, img_height),
                     format(width_sums), format(height_sums), format(area_sums), 
                     format(width_bars), format(height_bars), format(area_bars), 
                     format(area_interval), unit)