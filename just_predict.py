from ultralytics import YOLO
from os.path import join
from time import time

if __name__ == "__main__":
    model = YOLO("current_best_weight.pt")
    start = time()
    results = model.predict("predict_test.jpg", imgsz=512, conf=0.6)
    print(f"Demoró: {time() - start}")
    results[0].show()