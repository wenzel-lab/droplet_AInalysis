from os.path import join
from get_dimentions import get_dimentions
from PARAMETERS import PIXEL_RATIO, UNIT, IMGSZ, CONFIDENCE, TEST_IMAGE, TEST_WEIGHT, SAVE, MAX_DETECT
from get_boxes import get_boxes


decision = input("Choose what to do\n1. Get dimentions\n2. Show boxes\n3. Both\nX. Exit\n-> ")

if decision in "123":
    from ultralytics import YOLO as Yolo
    image_path = join("testing_imgs",TEST_IMAGE)
    model = Yolo(join("weights", TEST_WEIGHT))
    results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)

if decision == "1":
    image_info = get_dimentions(results, image_path, PIXEL_RATIO, UNIT)
    print(image_info)
elif decision == "2":
    get_boxes(results, image_path, TEST_IMAGE, TEST_WEIGHT, SAVE)
elif decision == "3":
    image_info = get_dimentions(results, image_path, PIXEL_RATIO, UNIT)
    print(image_info)
    get_boxes(results, image_path, TEST_IMAGE, TEST_WEIGHT, SAVE)
