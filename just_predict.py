from ultralytics import YOLO
import cv2
import numpy as np
from os.path import join
from time import time
from PARAMETERS import IMGSZ, CONFIDENCE


def just_predict(image_path, model, imgsz, conf, save=False):
    results = model.predict(image_path, imgsz = imgsz, conf=conf)
    image = cv2.imread(image_path)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save:
        cv2.imwrite(join("runs", "detect", image_path), image)

if __name__ == "__main__":
    image_path = "snapshot_2024-02-01_03-04-44.jpg"
    weights = "current_best_weights.pt"

    just_predict(image_path=image_path, model=YOLO(weights),imgsz=IMGSZ,conf=CONFIDENCE)