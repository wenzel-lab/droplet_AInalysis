import cv2
from threading import Thread, Lock, Event
from queue import Queue

from main_flow_functions import set_up, waiting_screen
from data_management.get_dimentions import get_dimentions
from data_management.get_boxes import get_boxes
from PARAMETERS import (PIXEL_RATIO, UNIT, IMGSZ, CONFIDENCE, 
                        MAX_DETECT, OMIT_BORDER_DROPLETS)


model_queue = Queue()
start_event = Event()

waiting_screen_thread = Thread(target=waiting_screen, args=(start_event,))
set_up_thread = Thread(target=set_up, args=(start_event, model_queue))

waiting_screen_thread.start()
set_up_thread.start()

waiting_screen_thread.join()
set_up_thread.join()

model = model_queue.get()