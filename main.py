from ultralytics import YOLO as Yolo
from get_dimentions import ImageParameters, get_dimentions

path = "predict_test.jpg"
weights = "current_best_weights.pt"
model = Yolo(weights)
image_info = get_dimentions(image_path=path, model=model, pixel_ratio=1)

print(image_info)
