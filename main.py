from os.path import join
from get_dimentions import get_dimentions
from PARAMETERS import (PIXEL_RATIO, UNIT, IMGSZ, 
                        CONFIDENCE, TEST_IMAGE, TEST_WEIGHT, 
                        SAVE, MAX_DETECT, OMIT_BORDER_DROPLETS)
from get_boxes import get_boxes
import matplotlib.pyplot as plot


decision = input("Choose what to do\n1. Get dimentions\n2. Show boxes\n3. Both\nX. Exit\n-> ")

if decision in "123":
    from ultralytics import YOLO as Yolo
    image_path = join("testing_imgs",TEST_IMAGE)
    model = Yolo(join("weights", TEST_WEIGHT))

if decision == "1":
    results = model.predict(join("testing_imgs", "snapshot_10.jpg"), imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)
    image_data1 = get_dimentions(results, join("testing_imgs", "snapshot_10.jpg"), PIXEL_RATIO, UNIT, OMIT_BORDER_DROPLETS)
    print(image_data1)
    results = model.predict(join("testing_imgs", "snapshot_56.jpg"), imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)
    image_data2 = get_dimentions(results, join("testing_imgs", "snapshot_56.jpg"), PIXEL_RATIO, UNIT, OMIT_BORDER_DROPLETS)
    print(image_data2)
    print(image_data1 + image_data2)
elif decision == "2":
    results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)
    get_boxes(results, image_path, TEST_IMAGE, TEST_WEIGHT, SAVE, OMIT_BORDER_DROPLETS)
elif decision == "3":
    results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)
    image_info = get_dimentions(results, image_path, PIXEL_RATIO, UNIT, OMIT_BORDER_DROPLETS)
    print(image_info)
    get_boxes(results, image_path, TEST_IMAGE, TEST_WEIGHT, SAVE, OMIT_BORDER_DROPLETS)
