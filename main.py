import cv2
from threading import Thread, Event
from queue import Queue

from main_flow_functions import (set_up, waiting_screen, manage_inputs,
                                 predict_v1, show_data)


model_queue = Queue()
start_event = Event()

waiting_screen_thread = Thread(target=waiting_screen, args=(start_event,))
set_up_thread = Thread(target=set_up, args=(start_event, model_queue))

waiting_screen_thread.start()
set_up_thread.start()

waiting_screen_thread.join()
set_up_thread.join()

model = model_queue.get()
image_data = model_queue.get() # empty instance of ImageParameters

events = {"pause": Event(), 
          "stop": Event(), 
          "forget": Event(), 
          "forgotten": Event(),
          "data_updated": Event()}

input_managing_thread = Thread(target=manage_inputs, args=(events,))
main_thread = Thread(target = predict_v1, args=(model, image_data, events))
show_data_thread = Thread(target=show_data, args=(image_data, events))