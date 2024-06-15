mode = input("\nWELCOME TO DROPLET AINALYSYS\n" +
             "Do you want to show Graphs?\n" + 
             "1. 'G' key to choose Graph mode\n" + 
             "2. Any other key to choose Terminal mode\n-> ")
import sys
sys.stdout.write("\033[F")
sys.stdout.write("\033[K")
sys.stdout.flush()
if mode == "G" or mode == "g":
    print("Graph mode has been chosen")
else:
    print("Terminal mode has been chosen")

from threading import Thread, Event
from queue import Queue
from copy import deepcopy
from backend import (manage_inputs, set_up, waiting_screen, graph_mode, terminal_mode)
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
if mode == "G" or mode == "g":
    backend_thread = Thread(target=graph_mode, 
                               args=(model, empty_data, deepcopy(empty_data), cap, events, main_queue), 
                               daemon=True)

    backend_thread.start()
    show_graphics(events, main_queue, PIXEL_RATIO)
    backend_thread.join()
else:
    frontend_thread = Thread(target=manage_inputs, 
                           args=(events,), 
                           daemon=True)
    backend_thread = Thread(target=terminal_mode, 
                               args=(model, empty_data, deepcopy(empty_data), cap, events), 
                               daemon=True)
    
    backend_thread.start()
    frontend_thread.start()
    backend_thread.join()
    frontend_thread.join()

print("Exit sucessfull!")