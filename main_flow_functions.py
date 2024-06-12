from time import sleep, time
import cv2
from os.path import join
from PARAMETERS import (PIXEL_RATIO, UNIT, IMGSZ, CONFIDENCE, 
                        MAX_DETECT, OMIT_BORDER_DROPLETS)
from data_management.get_dimentions import get_dimentions
from data_management.get_boxes import get_boxes
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

def manage_inputs(events):
    while not events["stop"].is_set():
        command = input("1. s to fully stop\n2. p to pause\n-> ")
        if command == "p" or command == "P":
            if events["pause"].is_set():
                events["pause"].clear()
            else:
                events["pause"].set()
        if command == "f":
            events["forget"].set()
        if command == "s" or command == "S":
            events["stop"].set()

        if events["forgotten"].is_set():
            events["forgotten"].clear()
            events["forget"].clear()

def add_to_queue(q, element):
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    q.put(element)       

def predict(model, image_data, cap, events, queue):
    while not events["stop"].is_set():
        if not events["pause"].is_set():
            if False:
                img = cv2.imread(join("testing_imgs","snapshot_00.jpg"))
            else:
                img = cap.read()[1]
            results = model.predict(img, imgsz = 640, conf=CONFIDENCE, max_det=MAX_DETECT)
            new_data = get_dimentions(results, PIXEL_RATIO, UNIT)
            image_data = image_data + new_data
            add_to_queue(queue, image_data)
            # opening the image, predicting and adding the data takes 18-35 miliseconds!
            # except for the first prediction which takes half a second
