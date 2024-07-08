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
                 diameter_sums: list, volume_sums: list,
                 diameter_bars: dict, volume_bars: dict, 
                 volume_interval: list,
                 unit: str, images_added = 1):

        self.n_droplets = n_droplets
        self.size = size

        self.diameter_sums = diameter_sums
        self.volume_sums = volume_sums

        self.diameter_bars = diameter_bars
        self.volume_bars = volume_bars

        self.volume_interval = volume_interval

        self.unit = unit
        self.images_added = images_added

        self._diameter_distribution = ["Diameter", "", 0, 1, self.unit, False] # [name, mean, stdd, unit, Calculated]
        self._volume_distribution = ["Volume", "", 0, 1, self.unit, False]

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
            if distribution[0] == "Volume" and -0.7<variance<0.7:
                variance = 0
            elif distribution[0]!= "Volume" and -0.01<variance<0.01:
                variance = 0

            distribution[2] = round(float(mean), rounding)
            distribution[3] = round(sqrt(variance), rounding)
        return distribution

    @property
    def diameter_distribution(self):
        return self._calculate_distribution(self.diameter_sums, self._diameter_distribution, 2)

    @property
    def volume_distribution(self):
        return self._calculate_distribution(self.volume_sums, self._volume_distribution, 2)

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

        new_diameter_bars = self._manage_addition(self.diameter_bars, other.diameter_bars, sum_bars, sub_bars)
        new_volume_bars = self._manage_addition(self.volume_bars, other.volume_bars, sum_bars, sub_bars)

        new_diameter_sums = self._manage_addition(self.diameter_sums, other.diameter_sums, sum_sums, sub_sums)
        new_volume_sums =  self._manage_addition(self.volume_sums, other.volume_sums, sum_sums, sub_sums)

        new_interval = self._manage_addition(self.volume_interval, other.volume_interval, sum_interval, sub_interval, "i")

        if ImageData.batch_size*ImageData.max_batches == self.images_added:
            self.images_added -= ImageData.batch_size

        images_added = self.images_added + 1

        return ImageData(new_n_droplets, self.size, 
                         new_diameter_sums, new_volume_sums, 
                         new_diameter_bars, new_volume_bars,
                         new_interval,
                         self.unit, images_added)

    def __str__(self):
        self._get_mode(self.diameter_bars, self.diameter_distribution)
        self._get_mode(self.volume_bars, self.volume_distribution)
        table_lines = tabulate(tabular_data=
                        [self.diameter_distribution[:5], 
                         self.volume_distribution[:5]], 
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
        