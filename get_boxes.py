from cv2 import imread, rectangle, imshow, waitKey, destroyAllWindows, imwrite
from os import chdir, path

def get_boxes(results, image_path, file_name, weight, save):
    image = imread(image_path)
    img_height, img_width, channels = image.shape
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if x1 != 0 and y1 != 0 and x2 < img_width-1 and y2 < img_height-1:
            rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    if save and not path.exists(file_name.split(".")[0] + "_" + weight.split("_")[1][:-3] + ".jpg"):
        chdir("saved_results")
        imwrite(file_name.split(".")[0] + "_" + weight.split("_")[1][:-3] + ".jpg", image)
    elif not save:
        imshow('Result', image)
        waitKey(0)
        destroyAllWindows()

if __name__ == "__main__":
    from ultralytics import YOLO as Yolo
    from PARAMETERS import IMGSZ, CONFIDENCE, TEST_IMAGE, TEST_WEIGHT, SAVE, MAX_DETECT

    file_name = TEST_IMAGE
    image_path = path.join("testing_imgs",file_name)

    weights = path.join("weights",TEST_WEIGHT)
    model = Yolo(path.join("weights", TEST_WEIGHT))
    results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)

    get_boxes(results, image_path, file_name, TEST_WEIGHT, SAVE)