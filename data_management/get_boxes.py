from PIL import Image, ImageDraw
from os import chdir, path

def get_boxes(results, image_path, file_name, weight, save, omit_border_droplets):
    image = Image.open(image_path)
    img_width, img_height = image.size

    draw = ImageDraw.Draw(image)

    droplet_images = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if (x1 > 1 and y1 > 1 and x2 < img_width-1 and y2 < img_height-1) or not omit_border_droplets:
            droplet_image = image.crop((x1, y1, x2, y2))

            droplet_images.append(droplet_image)

            if save:
                draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=1)
    if save:
        chdir(path.join("..", "saved_results"))
        image.save(file_name.split(".")[0] + "_" + weight.split("_")[1][:-3] + ".jpg")
    
    return droplet_images


if __name__ == "__main__":

    from ultralytics import YOLO as Yolo
    from PARAMETERS import IMGSZ, CONFIDENCE, TEST_IMAGE, TEST_WEIGHT, SAVE, MAX_DETECT, OMIT_BORDER_DROPLETS

    file_name = TEST_IMAGE
    image_path = path.join("testing_imgs",file_name)

    weights = path.join("weights",TEST_WEIGHT)
    model = Yolo(path.join("weights", TEST_WEIGHT))
    results = model.predict(image_path, imgsz = IMGSZ, conf=CONFIDENCE, max_det=MAX_DETECT)

    get_boxes(results, image_path, file_name, TEST_WEIGHT, SAVE, OMIT_BORDER_DROPLETS)