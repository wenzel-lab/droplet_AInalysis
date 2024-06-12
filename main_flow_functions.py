from time import sleep, time
from copy import deepcopy
import cv2
from os.path import join
from PARAMETERS import (PIXEL_RATIO, UNIT, IMGSZ, CONFIDENCE, 
                        MAX_DETECT)
from data_management.get_dimentions import get_dimentions
import queue


def waiting_screen(evento):
    i = 0
    while not evento.is_set():
        print(f"Seting up the model{'.'*i}", end='\r', flush=True)
        sleep(0.6)
        i+=1
    print("Model ready to predict!")

def set_up(evento, queue):
    from os.path import join
    from PARAMETERS import WEIGHT, IMGSZ, MAX_DETECT

    from ultralytics import YOLO as Yolo
    model = Yolo(WEIGHT)
    results = model.predict(join("testing_imgs","none.jpg"), imgsz=IMGSZ, max_det=MAX_DETECT, verbose=False)
    image_data = get_dimentions(results, PIXEL_RATIO, UNIT)
    cap = cv2.VideoCapture(0)
    queue.put(model)
    queue.put(image_data)
    queue.put(cap)

    evento.set()

def add_to_main_queue(q, element):
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    q.put(element)       

def predict(model, image_data, empty_data, cap, events, queue):
    while not events["exit"].is_set():
        if events["forget"].is_set():
            image_data = deepcopy(empty_data)
            events["forget"].clear()

        img = cap.read()[1]
        if not events["pause"].is_set():
            results = model.predict(img, imgsz = 640, conf=CONFIDENCE, max_det=MAX_DETECT)
            new_data = get_dimentions(results, PIXEL_RATIO, UNIT)
            image_data = image_data + new_data
        else:
            sleep(0.04)

        add_to_main_queue(queue, image_data)
