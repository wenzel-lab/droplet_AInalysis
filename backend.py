import sys
from time import sleep
from copy import deepcopy
import cv2
from PARAMETERS import (PIXEL_RATIO, UNIT, IMGSZ, CONFIDENCE, WEIGHT,  
                        MAX_DETECT)
from data_management.get_distributions import get_dimentions
import queue


def waiting_screen(evento):
    i = 0
    while not evento.is_set():
        print(f"Seting up the model{'.'*i}", end='\r', flush=True)
        sleep(0.6)
        i+=1
    print("                   " + " "*i)

def set_up(evento, queue):
    from os.path import join

    from ultralytics import YOLO as Yolo
    model = Yolo(WEIGHT)
    results = model.predict(join("imgs", "real_imgs","none.jpg"), imgsz=640, max_det=MAX_DETECT, verbose=False)
    image_data = get_dimentions(results, PIXEL_RATIO, UNIT)
    cap = cv2.VideoCapture(0)
    queue.put(model)
    queue.put(image_data)
    queue.put(cap)

    evento.set()   

def manage_inputs(events, extra1 = None, extra2 = None):
    first_loop = True
    while not events["exit"].is_set():
        if not first_loop:
            sys.stdout.write("\033[F" * 2)
            sys.stdout.write("\033[K\033[F" * 2 + "\033[K")
            sys.stdout.flush()
        command = input("1. e to Exit\n2. p to Pause or unPause\n3. f to Forget\n")

        if command == "p" or command == "P" or command.lower() == "pause" or command.lower() == "unpause":
            if events["pause"].is_set():
                events["pause"].clear()
            else:
                events["pause"].set()

        if command == "f" or command == "F" or command.lower() == "forget":
            events["forget"].set()

        if command == "e" or command == "E" or command.lower() == "exit":
            events["exit"].set()

        first_loop = False

def terminal_mode(model, image_data, empty_data, cap, events):
    sleep(0.01)
    print(image_data)
    image_counter = 1
    pause_frame = 0
    while not events["exit"].is_set():
        if events["forget"].is_set():
            image_data = deepcopy(empty_data)
            events["forget"].clear()
            image_counter = 1

        img = cap.read()[1]
        if not events["pause"].is_set():
            results = model.predict(img, imgsz = 640, conf=CONFIDENCE, max_det=MAX_DETECT)
            new_data = get_dimentions(results, PIXEL_RATIO, UNIT)
            image_data = image_data + new_data
            image_counter += 1
            pause_frame = 0
        else:
            pause_frame += 1

        if image_counter == 3 or (pause_frame == 3 and events["pause"].is_set()):
            sys.stdout.write("\033[F" * 5)
            sys.stdout.write("\033[K\033[F" * 4 + "\033[K")
            sys.stdout.flush()
            print(image_data)
            if image_counter ==3:
                image_counter = 0
            pause_frame = 0

def add_to_main_queue(q, element):
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    q.put(element)

def graph_mode(model, image_data, empty_data, cap, events, queue):
    while not events["exit"].is_set():
        if events["forget"].is_set():
            image_data = deepcopy(empty_data)
            events["forget"].clear()

        img = cap.read()[1]
        if not events["pause"].is_set():
            results = model.predict(img, imgsz = 640, conf=CONFIDENCE, max_det=MAX_DETECT)
            new_data = get_dimentions(results, PIXEL_RATIO, UNIT)
            image_data = image_data + new_data

        add_to_main_queue(queue, image_data)
