import cv2
from os import path

def get_boxes(results, img, file_name, weight, save):
    img_height, img_width = results[0].orig_shape

    droplet_images = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

            # droplet_image = img[y1:y2, x1:x2]

            # droplet_images.append(droplet_image)

        if save:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    print(4.5)
    if save:
        cv2.imwrite(path.join("saved_results", file_name.split(".")[0] + "_" + weight.split("_")[1][:-3] + ".jpg"), img)
    
    # return droplet_images