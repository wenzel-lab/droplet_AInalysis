from PIL import Image
from math import pi, sqrt
from os.path import join
from tabulate import tabulate

class ImageParameters:
    def __init__(self, n_droplets: int, xs: list, ys: list, widths: list, heights: list, pixel_ratio: float, unit="pixels"):
        self.n_droplets = n_droplets
        self.xs = xs
        self.ys = ys
        self.widths_lists = widths
        self.heights_lists = heights
        self.pixel_ratio = pixel_ratio
        self.unit = unit
        self._width = ["Width", 0, 0, self.unit] # name, mean, stdd, unit
        self._height = ["Height", 0, 0, self.unit]
        self._area = ["Area", 0, 0, self.unit]
    
    @property
    def width(self):
        if self._width[1] == self._width[2] == 0:
            mean = sum(w for w in self.widths_lists) / self.n_droplets
            variance = sum((w - mean)**2 for w in self.widths_lists) /self.n_droplets
            self._width[1] = mean
            self._width[2] = sqrt(variance)
        return self._width

    @property
    def height(self):
        if self._height[1] == self._height[2] == 0:
            mean = sum(h for h in self.heights_lists) / self.n_droplets
            variance = sum((h - mean)**2 for h in self.heights_lists) /self.n_droplets
            self._height[1] = mean
            self._height[2] = sqrt(variance)
        return self._height

    @property
    def area(self):
        if self._area[1] == self._area[2] == 0:
            total_area = sum((w * h) for w, h in zip(self.widths_lists, self.heights_lists))
            mean = total_area * pi * 0.25 / self.n_droplets
            variance = sum(((pi * w * h * 0.25) - mean) ** 2 for w, h in zip(self.widths_lists, self.heights_lists)) / self.n_droplets
            self._area[1] = mean
            self._area[2] = sqrt(variance)
        return self._area

    def __str__(self):
        return tabulate([self.width, self.height, self.area], 
                        headers=[str(self.n_droplets) + " Droplets", "Mean", "Std_dev", "Unit"], 
                        tablefmt="pretty")

def get_dimentions(results, image_path, pixel_ratio, unit):
    img_width, img_height = Image.open(image_path).size
    xs = []
    ys = []
    widths = []
    heights = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        if x1 != 0 and y1 != 0 and x2 < img_width-1 and y2 < img_height-1:
            width = x2 - x1
            height = y2 - y1

            xs.append(x1)
            ys.append(y1)
            widths.append(width)
            heights.append(height)
    return ImageParameters(len(xs), xs, ys, widths, heights, pixel_ratio, unit)

if __name__ == "__main__":
    from ultralytics import YOLO as Yolo
    from PARAMETERS import PIXEL_RATIO, UNIT, IMGSZ, CONFIDENCE, TEST_IMAGE, TEST_WEIGHT

    image_path = join("testing_imgs",TEST_IMAGE)
    model = Yolo(join("weights", TEST_WEIGHT))
    results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE)
    image_info = get_dimentions(results, image_path, PIXEL_RATIO, UNIT)
    print(image_info)