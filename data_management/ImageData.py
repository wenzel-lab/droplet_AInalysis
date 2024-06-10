from math import sqrt
from tabulate import tabulate
try:
    from data_tools import (sum_int, sub_int,
                            sum_sums, sub_sums,
                            sum_bars, sub_bars, 
                            sum_area_bars, sub_area_bars,
                            choose_interval)
except Exception:
    from data_management.data_tools import (sum_int, sub_int,
                                            sum_sums, sub_sums,
                                            sum_bars, sub_bars, 
                                            sum_area_bars, sub_area_bars,
                                            choose_interval)

class ImageData:
    batch_size = 60
    max_batches = 5

    def __init__(self, n_droplets: int, 
                 width_sums: list, height_sums: list, area_sums: list, 
                 width_bars: dict, height_bars: dict, area_bars: dict, 
                 area_interval: int, unit: str, images_added = 1):

        self.n_droplets = n_droplets

        self.width_sums = width_sums
        self.height_sums = height_sums
        self.area_sums = area_sums

        self.width_bars = width_bars
        self.height_bars = height_bars
        self.area_bars = area_bars
        self.area_interval = area_interval

        self.unit = unit
        self.images_added = images_added

        self._width_distribution = ["Width", 0, 1, self.unit, False] # [name, mean, stdd, unit, Calculated]
        self._height_distribution = ["Height", 0, 1, self.unit, False]
        self._area_distribution = ["Area", 0, 1, self.unit + "²", False]

    def _calculate_distribution(self, sums, distribution, rounding):
        calculated = distribution[4]
        if not calculated:
            distribution[4] = True

            mean = sums[0][0] / self.n_droplets[0] if self.n_droplets[0] else 0
            variance = sums[0][1] / self.n_droplets[0] - mean**2 if self.n_droplets[0] else 1

            distribution[1] = round(float(mean), rounding)
            distribution[2] = round(sqrt(variance), rounding)
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

    def _manage_addition(self, list1, list2, add, sub, chosen_interval = 0, chosen_instance = ""):
        if chosen_instance == "self":
            total = add(list1[0], list2[0], chosen_interval)
        elif chosen_instance == "other":
            total = add(list2[0], list1[0], chosen_interval)
        else:
            total = add(list1[0], list2[0])

        if self.images_added == ImageData.batch_size*ImageData.max_batches:
            if chosen_instance:
                total = sub(total, list1[1].pop(0), chosen_interval)
            else:
                total = sub(total, list1[1].pop(0))

        if not self.images_added%ImageData.batch_size:
            list1[1].append(list2[0])

        new_list = list1[1]
        if chosen_instance:
            new_list[-1] = add(list1[1][-1], list2[1][-1], chosen_interval)
        else:
            new_list[-1] = add(list1[1][-1], list2[1][-1])

        return [total, new_list]

    def __add__(self, other):
        new_n_droplets = self._manage_addition(self.n_droplets, other.n_droplets, sum_int, sub_int)

        new_area_interval, chosen_instance = choose_interval(self.area_interval, other.area_interval[0], self.images_added)

        new_width_bars = self._manage_addition(self.width_bars, other.width_bars, sum_bars, sub_bars)
        new_height_bars = self._manage_addition(self.height_bars, other.height_bars, sum_bars, sub_bars)
        new_area_bars = self._manage_addition(self.area_bars, other.area_bars, sum_area_bars, sub_area_bars, 
                                             new_area_interval[0], chosen_instance)

        new_width_sums = self._manage_addition(self.width_sums, other.width_sums, sum_sums, sub_sums)
        new_height_sums =  self._manage_addition(self.height_sums, other.height_sums, sum_sums, sub_sums)
        new_area_sums =  self._manage_addition(self.area_sums, other.area_sums, sum_sums, sub_sums)

        if ImageData.batch_size*ImageData.max_batches == self.images_added:
            self.images_added -= ImageData.batch_size

        images_added = self.images_added + 1

        return ImageData(new_n_droplets, new_width_sums, new_height_sums, new_area_sums, 
                         new_width_bars, new_height_bars, new_area_bars, new_area_interval,
                         self.unit, images_added)

    def __str__(self):
        return tabulate(tabular_data=
                        [self.width_distribution[:4], 
                         self.height_distribution[:4], 
                         self.area_distribution[:4]], 
                        headers=[str(self.n_droplets[0]) + " Droplets", "Mean", "Std_dev", "Unit"], 
                        tablefmt="pretty")