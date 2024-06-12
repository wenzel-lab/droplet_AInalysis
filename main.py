from threading import Thread, Event
from queue import Queue
from main_flow_functions import (set_up, waiting_screen, manage_inputs, predict)
from graphics import show_graphics
from PARAMETERS import PIXEL_RATIO


model_queue = Queue()
start_event = Event()

waiting_screen_thread = Thread(target=waiting_screen, args=(start_event,))
set_up_thread = Thread(target=set_up, args=(start_event, model_queue))

waiting_screen_thread.start()
set_up_thread.start()

waiting_screen_thread.join()
set_up_thread.join()

model = model_queue.get()
empty_data = model_queue.get() # empty instance of ImageData
cap = model_queue.get()

events = {"pause": Event(), 
          "stop": Event(), 
          "forget": Event(), 
          "forgotten": Event(),
          "data_updated": Event()}

main_queue = Queue()
inputs_thread = Thread(target=manage_inputs, args=(events,), daemon=True)
prediction_thread = Thread(target=predict, args=(model, empty_data, cap, events, main_queue), daemon=True)

inputs_thread.start()
prediction_thread.start()
show_graphics(events, main_queue, PIXEL_RATIO)

inputs_thread.join()
prediction_thread.join()