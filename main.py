from os.path import join
from get_dimentions import get_dimentions
from PARAMETERS import PIXEL_RATIO, UNIT, IMGSZ, CONFIDENCE, TEST_IMAGE, TEST_WEIGHT, SAVE
from show_boxes import show_boxes


decision = input("Choose what to do\n1. Get dimentions\n2. Show boxes\n3. Both\nX. Exit\n-> ")

if decision in "123":
    from ultralytics import YOLO as Yolo
    image_path = join("testing_imgs",TEST_IMAGE)
    model = Yolo(join("weights", TEST_WEIGHT))
    results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE)

if decision == "1":
    image_info = get_dimentions(results, image_path, PIXEL_RATIO, UNIT)
    print(image_info)
elif decision == "2":
    show_boxes(results, image_path, TEST_IMAGE, SAVE)
elif decision == "3":
    image_info = get_dimentions(results, image_path, PIXEL_RATIO, UNIT)
    print(image_info)
    show_boxes(results, image_path, TEST_IMAGE, SAVE)
