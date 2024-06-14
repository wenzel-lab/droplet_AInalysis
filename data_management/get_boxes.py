import cv2
from os import path

def get_boxes(results, img, file_name, weight):
    img_height, img_width = results[0].orig_shape
    # droplet_images = []
    array = results[0].boxes
    filter = (array.xyxy[:, 0] > 1) & (array.xyxy[:, 1] > 1) & (array.xyxy[:, 2] < img_width - 1) & (array.xyxy[:, 3] < img_height - 1)
    array = array[filter]

    for box in array:
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # droplet_image = img[y1:y2, x1:x2]
        # droplet_images.append(droplet_image)
        cv2.ellipse(img, 
                    center=(((x1 + x2) // 2, (y1 + y2) // 2)), 
                    axes=((x2 - x1)//2, (y2- y1)//2), 
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=(0, 255, 0), 
                    thickness=1)

    cv2.putText(img, weight.split("_")[1][:-3], (7,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    img_path = path.join("imgs", "results", file_name.split(".")[0], "latest_prediction_" + weight.split("_")[1][:-3] + ".jpg")
    cv2.imwrite(img_path, img)
    return img_path

    # return droplet_images