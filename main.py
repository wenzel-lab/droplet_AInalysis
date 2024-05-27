from ultralytics import YOLO as Yolo
from get_dimentions import ImageParameters, get_dimentions

path = "predict_test.jpg"
model = Yolo("current_best_weights.pt")
image_info = get_dimentions(image_path=path, model=model, pixel_ratio=1)

print(image_info)