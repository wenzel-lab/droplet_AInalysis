from math import sqrt
from tabulate import tabulate
from PARAMETERS import PIXEL_RATIO
from data_management.data_tools import (sum_int, sub_int,
                                        sum_interval, sub_interval,
                                        sum_sums, sub_sums,
                                        sum_bars, sub_bars, 
                                        group_in_intervals)

class ImageData:
    batch_size = 60
    max_batches = 5

    def __init__(self, n_droplets: int, size: tuple,
                 width_sums: list, height_sums: list, area_sums: list, 
                 width_bars: dict, height_bars: dict, area_bars: dict, 
                 area_interval: int, unit: str, images_added = 1):

        self.n_droplets = n_droplets
        self.size = size

        self.width_sums = width_sums
        self.height_sums = height_sums
        self.area_sums = area_sums

        self.width_bars = width_bars
        self.height_bars = height_bars
        self.area_bars = area_bars
        self.area_interval = area_interval

        self.unit = unit
        self.images_added = images_added

        self._width_distribution = ["Width", "", 0, 1, self.unit, False] # [name, mean, stdd, unit, Calculated]
        self._height_distribution = ["Height", "", 0, 1, self.unit, False]
        self._area_distribution = ["Area", "", 0, 1, self.unit + "²", False]

    def _get_mode(self, bars, distribution, interval = 1):
        if interval != 1:
            bars_to_use = group_in_intervals(bars[0], interval[0])
        else:
            bars_to_use = bars[0]

        maximum = 0
        mode = 0
        for element, quantity in bars_to_use:
            if quantity > maximum:
                maximum = quantity
                mode = element

        pixel_ratio = PIXEL_RATIO
        string_interval = ""
        if distribution[0] == "Area":
            pixel_ratio *= PIXEL_RATIO
            string_interval = f" ± {round(interval[0]*pixel_ratio*0.5, 2)}"
        mode = round(mode * pixel_ratio, 2)
        distribution[1] = f"{mode}{string_interval}"         

    def _calculate_distribution(self, sums, distribution, rounding):
        calculated = distribution[5]
        if not calculated:
            distribution[5] = True
            mean = sums[0][0] / self.n_droplets[0] if self.n_droplets[0] else 0
            variance = sums[0][1] / self.n_droplets[0] - mean**2 if self.n_droplets[0] else 1
            if distribution[0] == "Area" and -0.1<variance<0.1:
                variance = 0
            elif distribution[0]!= "Area" and -0.01<variance<0.01:
                variance = 0

            distribution[2] = round(float(mean), rounding)
            distribution[3] = round(sqrt(variance), rounding)
        return distribution

    @property
    def width_distribution(self):
        return self._calculate_distribution(self.width_sums, self._width_distribution, 2)

    @property
    def height_distribution(self):
        return self._calculate_distribution(self.height_sums, self._height_distribution, 2)

    @property
    def area_distribution(self):
        return self._calculate_distribution(self.area_sums, self._area_distribution, 2)

    def _manage_addition(self, list1, list2, add, sub, interval = ""):
        total = add(list1[0], list2[0])
        if self.images_added == ImageData.batch_size*ImageData.max_batches:
            removed_element = list1[1].pop(0)
            if interval == "i":
                total = sub(total, list1[1])
            else:
                total = sub(total, removed_element)

        new_list = list1[1]
        if not self.images_added%ImageData.batch_size:
            list1[1] = list1[1].append(list2[0])
        else:
            new_list[-1] = add(list1[1][-1], list2[1][-1])

        return [total, new_list]

    def __add__(self, other):
        new_n_droplets = self._manage_addition(self.n_droplets, other.n_droplets, sum_int, sub_int)

        new_area_interval = self._manage_addition(self.area_interval, other.area_interval, sum_interval, sub_interval, "i")

        new_width_bars = self._manage_addition(self.width_bars, other.width_bars, sum_bars, sub_bars)
        new_height_bars = self._manage_addition(self.height_bars, other.height_bars, sum_bars, sub_bars)
        new_area_bars = self._manage_addition(self.area_bars, other.area_bars, sum_bars, sub_bars)

        new_width_sums = self._manage_addition(self.width_sums, other.width_sums, sum_sums, sub_sums)
        new_height_sums =  self._manage_addition(self.height_sums, other.height_sums, sum_sums, sub_sums)
        new_area_sums =  self._manage_addition(self.area_sums, other.area_sums, sum_sums, sub_sums)

        if ImageData.batch_size*ImageData.max_batches == self.images_added:
            self.images_added -= ImageData.batch_size

        images_added = self.images_added + 1

        return ImageData(new_n_droplets, self.size, 
                         new_width_sums, new_height_sums, new_area_sums, 
                         new_width_bars, new_height_bars, new_area_bars, new_area_interval,
                         self.unit, images_added)

    def __str__(self):
        self._get_mode(self.width_bars, self.width_distribution)
        self._get_mode(self.height_bars, self.height_distribution)
        self._get_mode(self.area_bars, self.area_distribution, self.area_interval)
        table_lines = tabulate(tabular_data=
                        [self.width_distribution[:5], 
                         self.height_distribution[:5], 
                         self.area_distribution[:5]], 
                        headers=[str(round(self.n_droplets[0]/self.images_added, 2)) + " Droplets/image", 
                                 "Mode", 
                                 "Mean", 
                                 "Std Dev", 
                                 "Unit"], 
                        tablefmt="pretty").split("\n")

        return_string = ""

        ammount_string = f"| {self.images_added} {'images' if self.images_added > 1 else 'image'} |\n"
        return_string += f"+{'-'*(len(ammount_string)-3)}+\n"
        return_string += ammount_string
        for line in table_lines:
            return_string += line + "        \n"

        return return_string.rstrip("\n")
        