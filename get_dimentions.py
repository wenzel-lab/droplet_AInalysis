from ultralytics import YOLO
from math import pi, sqrt

class ImageParameters:
    def __init__(self, number_of_droplets: int, xs: list, ys: list, widths: list, heights: list, pixel_ratio: float, unit="pixels"):
        self.number_of_droplets = number_of_droplets
        self.xs = xs
        self.ys = ys
        self.widths = widths
        self.heights = heights
        self.pixel_ratio = pixel_ratio
        self.unit = unit
        self._mean_area = None
        self._std_dev_area = None

    @property
    def mean_area(self):
        if self._mean_area is None:
            total_area = sum(pi * w * h for w, h in zip(self.widths, self.heights))
            self._mean_area = total_area / self.number_of_droplets
        return self._mean_area
    
    @property
    def std_dev_area(self):
        if self._std_dev_area is None:
            if self._mean_area is None:
                _ = self.mean_area
            mean = self._mean_area
            variance = sum((pi * w * h - mean) ** 2 for w, h in zip(self.widths, self.heights)) / self.number_of_droplets
            self._std_dev_area = sqrt(variance)
        return self._std_dev_area
    
    def __str__(self):
        return (f"\n{self.number_of_droplets} droplets detected\n" +
                f"Mean area is {self.mean_area * self.pixel_ratio} {self.unit}\n" +
                f"Standard deviation of area is {self.std_dev_area * self.pixel_ratio} {self.unit}")
    

def get_dimentions(image_path, model, pixel_ratio=1, unit="pixels"):
    result = model.predict(image_path, imgsz=1024, conf=0.6)
    xs = []
    ys = []
    widths = []
    heights = []
    for box in result[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        width = x2 - x1
        height = y2 - y1

        xs.append(x1)
        ys.append(y1)
        widths.append(width)
        heights.append(height)
    return ImageParameters(len(xs), xs, ys, widths, heights, pixel_ratio, unit)
    
    




