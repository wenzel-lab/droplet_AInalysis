from ultralytics import YOLO as Yolo
from get_dimentions import get_dimentions

path = "snapshot_2024-02-01_03-04-44.jpg"
model = Yolo("current_best_weights.pt")
image_info = get_dimentions(image_path=path, model=model, pixel_ratio=1)

print(image_info)
