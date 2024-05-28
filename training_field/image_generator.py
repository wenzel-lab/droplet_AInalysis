from random import randint
from PIL import Image

def check_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    return not (x1 + w1 <= (x2-3) or x2 + w2 <= (x1-3) or y1 + h1 <= (y2-3) or y2 + h2 <= (y1-3))

def place_images(big_image_path, small_images_path, num_small_images, output_image_path, output_labels_path):
    big_image = Image.open(big_image_path)
    small_images = [Image.open(i) for i in small_images_path]
    
    big_width, big_height = big_image.size
    smalls_dimentions = [i.size for i in small_images]

    labels = []
    positions = []

    combined_image = big_image.copy()
    
    max_attempts = 100
    for _ in range(num_small_images):
        chosen_small = randint(0, len(small_images)-1)
        small_image = small_images[chosen_small]
        small_width, small_height = smalls_dimentions[chosen_small]

        attempts = 0
        while attempts < max_attempts:
            x = randint(small_width, big_width)
            y = randint(small_height, big_height)

            overlap = False
            for (px, py, pwidth, pheight) in positions:
                if check_overlap(x, y, small_width, small_height, px, py, pwidth, pheight):
                    overlap = True
                    break
            
            if not overlap:
                combined_image.paste(small_image, (x, y), small_image)

                x_center = (x + small_width / 2) / big_width
                y_center = (y + small_height / 2) / big_height
                width = small_width / big_width
                height = small_height / big_height

                labels.append(f"0 {x_center} {y_center} {width} {height}")
                positions.append((x, y, small_width, small_height))
                break
            
            attempts += 1

    combined_image.save(output_image_path)

    with open(output_labels_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")