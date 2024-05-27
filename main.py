from ultralytics import YOLO as Yolo
from get_dimentions import get_dimentions
from PARAMETERS import PIXEL_RATIO, UNIT, IMGSZ, CONFIDENCE

path = "snapshot_2024-02-01_03-04-44.jpg"
model = Yolo("current_best_weights.pt")
image_info = get_dimentions(image_path=path, model=model, pixel_ratio=PIXEL_RATIO, unit=UNIT, imgsz=IMGSZ, conf=CONFIDENCE)

print(image_info)