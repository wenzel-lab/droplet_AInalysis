from random import randint, choices, choice
from PIL import Image
from image_tools import (random_darkening, transparency, 
                         check_overlap, crop, rotate, 
                         flip_flop, expand, color_filters,
                         blur)


def format_coordinates(x, y, small_width, small_height, big_width, big_height) -> str:
    x_center = (x + small_width / 2) / big_width
    y_center = (y + small_height / 2) / big_height
    width = small_width / big_width
    height = small_height / big_height

    return f"0 {x_center} {y_center} {width} {height}"

def place_images(big_image_path, small_images_path, num_small_images, output_image_path, output_labels_path):
    big_image = crop(Image.open(big_image_path), size=640)
    big_image = rotate(big_image)
    big_image = flip_flop(big_image)
    big_width, big_height = big_image.size

    darken = choices(["background", "final", "none"], weights=[0.2, 0.5, 0.3], k=1)[0]
    if darken == "background":
        big_image = random_darkening(big_image)

    small_images = [Image.open(i).convert("RGBA") for i in small_images_path]

    labels = []
    positions = []

    combined_image = big_image.copy()

    max_attempts = 25
    n_placed = failed = 0
    while n_placed < num_small_images and failed < 25:
        small_image = choice(small_images)
        small_image = rotate(small_image)
        small_image = flip_flop(small_image)
        small_image = expand(small_image)
        small_image = color_filters(small_image)
        small_image = random_darkening(small_image)
        small_image = transparency(small_image)
        small_image = blur(small_image)
        small_width, small_height = small_image.size

        attempts = 0
        placed = False
        while attempts < max_attempts and not placed:
            x = randint(0, big_width-small_width)
            y = randint(0, big_height-small_height)

            overlap = check_overlap(x, y, small_width, small_height, positions)

            if not overlap:
                combined_image.paste(small_image, (x, y), small_image)

                labels.append(format_coordinates(x, y, small_width, small_height, big_width, big_height))
                positions.append((x, y, small_width, small_height))

                placed = True
                n_placed += 1
            else:
                attempts += 1

        if attempts == max_attempts:
            failed += 1

    if darken == "final":
        combined_image = random_darkening(combined_image)

    combined_image.save(output_image_path)

    with open(output_labels_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
