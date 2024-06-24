from random import uniform, choice, randint, choices, random
from math import ceil
from PIL import ImageEnhance, Image, ImageFilter


def check_overlap(x1, y1, w1, h1, positions):
    o = 3 # Increase this value for more overlap of droplets
    for (x2, y2, w2, h2) in positions:
        if (x2 + o < x1 + w1 and x1 < x2 + w2 - o) and (y2 + o < y1 + h1 and y1 < y2 + h2 - o):
            return True
    return False

def rotate(image):
    rotate = choice([0, 90, 180, 270])

    return image.rotate(rotate, expand=True)

def flip_flop(image):
    flip = choice([True, False])
    flop = choice([True, False])
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if flop:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image

def expand(image):
    width, height = image.size
    expand_width = choices([True, False], weights=[0.3, 0.7], k=1)[0]
    expand_height = choices([True, False], weights=[0.3, 0.7], k=1)[0]

    if expand_width:
        width += randint(-width//3, width//3)
    if expand_height:
        height += randint(-height//3,height//3)

    return image.resize((width, height))

def random_darkening(image):
    factor = uniform(0.3, 1.5)
    enhancer = ImageEnhance.Brightness(image)
    darken_image = enhancer.enhance(factor)

    return darken_image

def crop(image, size: int):
    current_width, current_height = image.size
    left = (current_width - size) / 2
    top = (current_height - size) / 2
    right = (current_width + size) / 2
    bottom = (current_height + size) / 2
    return image.crop((left, top, right, bottom))

def color_filters(image):
    width, height = image.size
    pixels_RGBA = image.getdata()

    new_data = []
    apply = choices(["blue", "green", "both", ""], weights=[0.1, 0.1, 0.1, 0.7], k=1)[0]

    y = 0
    while apply and y < height:
        for x in range(width):
            r, g, b, a = pixels_RGBA[y * width + x]

            if apply == "blue":
                new_data.append((b, g, r, a))
            if apply == "green":
                new_data.append((g, r, b, a))
            if apply == "both":
                new_data.append((b, r, r, a))
        y += 1
    new_image = Image.new("RGBA", (width, height))
    new_image.putdata(new_data)

    if apply:
        return new_image
    else:
        return image

def transparency(image):
    start = choice(["left", "right", "top", "bottom"])
    width, height = image.size
    pixels_RGBA = image.getdata()

    new_data = []
    apply_gradient = choices([True, False], weights=[0.4, 0.6], k=1)[0]
    starting_transparency = randint(10, 110)
    gradient_size = uniform(0.1, 0.7) # effect apllied between 0.1 and 0.7 of the image

    y = 0
    while apply_gradient and y < height:
        for x in range(width):
            r, g, b, a = pixels_RGBA[y * width + x]

            if start == "left" and x < width * gradient_size and a > 50:
                a = starting_transparency + int((255 - starting_transparency) * (x / (width * gradient_size)))
            if start == "right" and x >= width * (1 - gradient_size) and a > 50:
                a = starting_transparency + int((255 - starting_transparency) * ((width - 1 - x) / (width * gradient_size)))
            if start == "top" and y < height * gradient_size and a > 50:
                a = starting_transparency + int((255 - starting_transparency) * (y / (height * gradient_size)))
            if start == "bottom" and y >= height * (1-gradient_size) and a > 50:
                a = starting_transparency + int((255 - starting_transparency) * ((height - 1 - y) / (height*gradient_size)))

            new_data.append((r, g, b, a))
        y += 1

    new_image = Image.new("RGBA", (width, height))
    new_image.putdata(new_data)

    if apply_gradient:
        return new_image
    return image

def blur(image):
    small_width, small_height = image.size
    apply_effect = choices([False, True], weights=[0.3, 0.7])
    if apply_effect:
        blur_radius = random()*2
        margin = ceil(blur_radius)

        new_width = small_width + 2 * margin
        new_height = small_height + 2 * margin

        new_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
        new_image.paste(image, (margin, margin))

        return new_image.filter(ImageFilter.GaussianBlur(radius = blur_radius))
    return image