from ultralytics import YOLO as Yolo
from os.path import join
from get_dimentions import get_dimentions
from PARAMETERS import PIXEL_RATIO, UNIT, IMGSZ, CONFIDENCE

path = join("testing_imgs","snapshot_44.jpg")
model = Yolo("best.pt")
image_info = get_dimentions(image_path=path, model=model, pixel_ratio=PIXEL_RATIO, unit=UNIT, imgsz=IMGSZ, conf=CONFIDENCE)

print(image_info)