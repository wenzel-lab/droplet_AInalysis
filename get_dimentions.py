from PIL import Image
from math import pi, sqrt
from os.path import join
from tabulate import tabulate
from data_tools import sum_dictionarys

class ImageParameters:
    def __init__(self, n_droplets: int, 
                 widths: list, heights: list, areas: list, 
                 width_bars: dict, height_bars: dict, area_bars: dict,
                 pixel_ratio: float, unit: str,
                 width_sums, height_sums, area_sums):

        self.n_droplets = n_droplets

        self.widths = widths
        self.heights = heights
        self.areas = areas

        self.width_bars = width_bars
        self.height_bars = height_bars
        self.area_bars = area_bars

        self.pixel_ratio = pixel_ratio
        self.unit = unit

        self._width_sums = width_sums # list: [sum, sum of squares]
        self._height_sums = height_sums
        self._area_sums = area_sums

        self._width_distribution = ["Width", 0, 1, self.unit, False] # [name, mean, stdd, unit, Calculated]
        self._height_distribution = ["Height", 0, 1, self.unit, False]
        self._area_distribution = ["Area", 0, 1, self.unit + "²", False]

    def _calculate_sums(self, data, sums):
        while data:
            value = data.pop(0)
            sums[0] += value
            sums[1] += value**2
        return sums

    @property
    def width_sums(self):
        return self._calculate_sums(self.widths, self._width_sums)
    
    @property
    def height_sums(self):
        return self._calculate_sums(self.heights, self._height_sums)

    @property
    def area_sums(self):
        return self._calculate_sums(self.areas, self._area_sums)

    def _calculate_distribution(self, sums, distribution, rounding):
        calculated = distribution[4]
        if not calculated:
            distribution[4] = True

            mean = sums[0] / self.n_droplets if self.n_droplets else 0
            variance = sums[1] / self.n_droplets - mean**2 if self.n_droplets else 1

            distribution[1] = round(float(mean * self.pixel_ratio), rounding)
            distribution[2] = round(sqrt(variance) * self.pixel_ratio, rounding)
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

    def __add__(self, other):
        n_droplets = self.n_droplets + other.n_droplets

        return ImageParameters(n_droplets, [], [], [], 
                               sum_dictionarys(self.width_bars, other.width_bars), 
                               sum_dictionarys(self.height_bars, other.height_bars), 
                               sum_dictionarys(self.area_bars, other.area_bars), 
                               self.pixel_ratio, self.unit, 
                               [w1 + w2 for w1, w2 in zip(self.width_sums, other.width_sums)], 
                               [h1 + h2 for h1, h2 in zip(self.height_sums, other.height_sums)], 
                               [a1 + a2 for a1, a2 in zip(self.area_sums, other.area_sums)])

    def __str__(self):
        return tabulate(tabular_data=
                        [self.width_distribution[:4], 
                         self.height_distribution[:4], 
                         self.area_distribution[:4]], 
                        headers=[str(self.n_droplets) + " Droplets", "Mean", "Std_dev", "Unit"], 
                        tablefmt="pretty")

def get_dimentions(results, image_path, pixel_ratio, unit, omit_border_droplets):
    img_width, img_height = Image.open(image_path).size
    widths = []
    heights = []
    areas = []
    width_bars = {}
    height_bars = {}
    area_bars = {}
    ellipse_constant = pi * 0.25

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        if (x1 > 1 and y1 > 1 and x2 < img_width-1 and y2 < img_height-1) or not omit_border_droplets:
            width = float(x2 - x1)
            height = float(y2 - y1)
            area = float(height * width * ellipse_constant)

            if width_bars.get(round(width), -1) > 0:
                width_bars[round(width)] += 1
            else:
                width_bars[round(width)] = 1
            if height_bars.get(round(height), -1) > 0:
                height_bars[round(height)] += 1
            else:
                height_bars[round(height)] = 1
            if area_bars.get(round(area), -1) > 0:
                area_bars[round(area)] += 1
            else:
                area_bars[round(area)] = 1

            widths.append(width)
            heights.append(height)
            areas.append(area)
    return ImageParameters(len(widths), widths, heights, areas, 
                           width_bars, height_bars, area_bars, 
                           pixel_ratio, unit,
                           [0, 0], [0, 0], [0, 0])

if __name__ == "__main__":
    from ultralytics import YOLO as Yolo
    from PARAMETERS import PIXEL_RATIO, UNIT, IMGSZ, CONFIDENCE, TEST_IMAGE, TEST_WEIGHT, MAX_DETECT, OMIT_BORDER_DROPLETS

    image_path = join("testing_imgs",TEST_IMAGE)
    model = Yolo(join("weights", TEST_WEIGHT))
    results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)
    image_info = get_dimentions(results, image_path, PIXEL_RATIO, UNIT, OMIT_BORDER_DROPLETS)
    print(image_info)