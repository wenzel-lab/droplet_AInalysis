IMGSZ = 1024 # size of the image that is analysed
CONFIDENCE = 0.51 # minimum confidence that a prediction needs to be counted as a droplet
MAX_DETECT = 500 # maximum amount of droplets that can be detected in the image
PIXEL_RATIO = 1
UNIT = "pixels"
TEST_IMAGE = "snapshot_58.jpg"
TEST_WEIGHT = "best_82.pt"
OMIT_BORDER_DROPLETS = True
SAVE = True # only for show_boxes.py