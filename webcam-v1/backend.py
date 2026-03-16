import sys
from time import sleep
from copy import deepcopy
import cv2
from PARAMETERS import (PIXEL_RATIO, UNIT, IMGSZ, CONFIDENCE, WEIGHT,  
                        MAX_DETECT)


def live_view_mode(model, cap, events):
    import cv2
    while not events["exit"].is_set():
        ret, img = cap.read()
        if not ret:
            break
        results = model.predict(img, imgsz=640, conf=CONFIDENCE, max_det=MAX_DETECT)
        # Draw bounding boxes
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Live Droplet Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            events["exit"].set()
            break
    cv2.destroyAllWindows()

import os
from data_management.get_distributions import get_dimensions
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
    
    # Load weight from webcam-v1 local weights folder
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    weights_path = os.path.join(weights_dir, WEIGHT)
    model = Yolo(weights_path)
    results = model.predict(join("imgs", "real_imgs","none.jpg"), imgsz=640, max_det=MAX_DETECT, verbose=False)
    image_data = get_dimensions(results, PIXEL_RATIO, UNIT)
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
            new_data = get_dimensions(results, PIXEL_RATIO, UNIT)
            image_data = image_data + new_data
            image_counter += 1
            pause_frame = 0

            # --- Live annotated view ---
            # Draw bounding boxes on the frame
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Live Droplet Detection", img)
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                events["exit"].set()
                break
        else:
            pause_frame += 1

        if image_counter == 10 or (pause_frame == 10 and events["pause"].is_set()):
            sys.stdout.write("\033[F" * 4)
            sys.stdout.write("\033[K\033[F" * 4 + "\033[K")
            sys.stdout.flush()
            print(image_data)
            if image_counter == 10:
                image_counter = 0
            pause_frame = 0

    # Clean up OpenCV window on exit
    cv2.destroyAllWindows()

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
            new_data = get_dimensions(results, PIXEL_RATIO, UNIT)
            image_data = image_data + new_data

        add_to_main_queue(queue, image_data)
