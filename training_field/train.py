from ultralytics import YOLO
from os.path import join
import torch
torch.cuda.empty_cache()
torch.device("cuda:0")


if __name__ == "__main__":
    model = YOLO('yolov8n.yaml').load(join("runs","detect","test26","weights","best.pt"))

    results = model.train(data=join("yamls","train_1.yaml"), epochs=10, workers=2, imgsz=1024, batch=3, name="test3", device=0, close_mosaic=0)