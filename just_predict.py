from ultralytics import YOLO
import cv2
import numpy as np
from os.path import join
from time import time

image_path = "predict_test.jpg"
weights = "current_best_weights.pt"
confidence = 0.5

model = YOLO(weights)
start = time()
results = model.predict(image_path, imgsz=1024, conf=confidence, device=0)
print(f"Demoró: {time() - start}")

image = cv2.imread(image_path)
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save image
# cv2.imwrite(join("runs", "detect", image_path), image)