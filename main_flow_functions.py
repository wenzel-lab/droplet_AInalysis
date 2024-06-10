from time import sleep
from scipy import stats
import numpy as np
import cv2
from matplotlib.pyplot import plt as plot
from PARAMETERS import (PIXEL_RATIO, UNIT, IMGSZ, CONFIDENCE, 
                        MAX_DETECT, OMIT_BORDER_DROPLETS)
from data_management.get_dimentions import get_dimentions
from data_management.get_boxes import get_boxes

def waiting_screen(evento):
    i = 0
    while not evento.is_set():
        print(f"Seting up the model{'.'*i}", end='\r', flush=True)
        sleep(0.8)
        i+=1
    print("Model ready to predict!")

def set_up(evento, queue):
    from os.path import join
    from ultralytics import YOLO as Yolo
    from PARAMETERS import WEIGHT, IMGSZ, MAX_DETECT

    model = Yolo(WEIGHT)
    results = model.predict(join("testing_imgs","none.jpg"), imgsz=IMGSZ, max_det=MAX_DETECT)
    image_data = get_dimentions(results, join("testing_imgs","none.jpg"), PIXEL_RATIO, UNIT, OMIT_BORDER_DROPLETS)
    queue.put(model)
    queue.put(image_data)

    evento.set()
    return model

def manage_inputs(events):
    while not events["stop"].is_set():
        command = input()
        if command == "":
            if events["pause"].is_set():
                events["pause"].clear()
            else:
                events["pause"].set()
        if command == "f":
            events["forget"].set()
        if command == "s":
            events["stop"].set()

        if events["forgotten"].is_set():
            events["forgotten"].clear()
            events["forget"].clear()

def predict_v1(model, image_data, events):
    while not events["stop"].is_set():
        if not events["pause"].is_set():
            cap = cv2.VideoCapture(0)
            frame = cap.read()[1]
            cap.release()

            image_path = "current_image.jpg"
            cv2.imwrite(image_path, frame)

            results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)
            image_data += get_dimentions(results, image_path, PIXEL_RATIO, UNIT, OMIT_BORDER_DROPLETS)
            events["data_updated"].set()
            

def predict_v2():
    pass