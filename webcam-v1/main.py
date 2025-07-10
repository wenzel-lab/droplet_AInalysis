mode = input("\nWELCOME TO DROPLET AINALYSYS\n" +
             "Choose a mode:\n" + 
             "1. 'G' key for Graph mode (plots/statistics)\n" + 
             "2. 'L' key for Live View (annotated webcam only)\n" + 
             "3. 'B' key for Both Live View + Statistics\n" + 
             "4. Any other key for Terminal mode\n-> ")
import sys
sys.stdout.write("\033[F")
sys.stdout.write("\033[K")
sys.stdout.flush()
if mode == "G" or mode == "g":
    print("Graph mode has been chosen")
elif mode == "L" or mode == "l":
    print("Live View mode has been chosen")
elif mode == "B" or mode == "b":
    print("Both Live View + Statistics mode has been chosen")
else:
    print("Terminal mode has been chosen")

from threading import Thread, Event
from queue import Queue
from copy import deepcopy
from backend import (manage_inputs, set_up, waiting_screen, graph_mode, terminal_mode, live_view_mode)
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
    # Create a thread for backend analysis in graph mode
    backend_thread = Thread(target=graph_mode, 
                               args=(model, empty_data, deepcopy(empty_data), cap, events, main_queue), 
                               daemon=True)

    # Start the backend analysis thread for graph mode
    backend_thread.start()
    # Show the live matplotlib GUI for real-time statistics
    show_graphics(events, main_queue, PIXEL_RATIO)
    # Wait for backend analysis to finish before exiting
    backend_thread.join()
elif mode == "L" or mode == "l":
    # Live View only (no statistics)
    live_thread = Thread(target=live_view_mode,
                        args=(model, cap, events),
                        daemon=True)
    live_thread.start()
    live_thread.join()
elif mode == "B" or mode == "b":
    # Unified UI: Live View + Statistics in one window
    backend_thread = Thread(target=graph_mode, 
                               args=(model, empty_data, deepcopy(empty_data), cap, events, main_queue), 
                               daemon=True)
    backend_thread.start()
    from frontend import show_combined_ui
    show_combined_ui(model, cap, events, main_queue, PIXEL_RATIO)
    backend_thread.join()
else:
    # In terminal mode, start the thread for user input (pause, reset, exit)
    frontend_thread = Thread(target=manage_inputs, 
                           args=(events,), 
                           daemon=True)
    # Start the backend thread for terminal-based analysis
    backend_thread = Thread(target=terminal_mode, 
                               args=(model, empty_data, deepcopy(empty_data), cap, events), 
                               daemon=True)
    
    # Start both threads
    backend_thread.start()
    frontend_thread.start()
    # Wait for both threads to finish before exiting
    backend_thread.join()
    frontend_thread.join()

print("Exit successful!")