from ultralytics import YOLO
from os.path import join
import torch
torch.cuda.empty_cache()
torch.device("cuda:0")


if __name__ == "__main__":
    model = YOLO('yolov8n.yaml').load(join("runs","detect","test54","weights","best.pt"))

    results = model.train(data="train_1.yaml", epochs=52, workers=2, imgsz=640, batch=13, name="test7", device=0, close_mosaic=0)