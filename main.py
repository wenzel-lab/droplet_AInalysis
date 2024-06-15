mode = input('Do you want to show Graphs?\n1. "Y" key to show graphs\n2. Any other key to not show graphs\n-> ')

from threading import Thread, Event
from queue import Queue
from copy import deepcopy
from backend import (manage_inputs, set_up, waiting_screen, predict)
from frontend import show_graphics
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

events = {"exit": Event(),
          "pause": Event(), 
          "forget": Event()}

main_queue = Queue()
if mode == "Y" or mode == "y":
    graphics_thread = Thread(target=show_graphics, args=(events, main_queue, PIXEL_RATIO), daemon=True)
    prediction_thread = Thread(target=predict, args=(model, empty_data, deepcopy(empty_data), cap, events, main_queue, True), daemon=True)

    prediction_thread.start()
    graphics_thread.start()

    prediction_thread.join()
    graphics_thread.join()
else:
    inputs_thread = Thread(target=manage_inputs, args=(events,), daemon=True)
    prediction_thread = Thread(target=predict, args=(model, empty_data, deepcopy(empty_data), cap, events, main_queue, False), daemon=True)
    
    prediction_thread.start()
    inputs_thread.start()

    prediction_thread.join()
    inputs_thread.join()

print("Exit sucessfull!")