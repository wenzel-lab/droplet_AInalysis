from random import uniform

from PIL import ImageEnhance

def random_darkening(imagen):
    factor = uniform(0.6, 0.85)

    enhancer = ImageEnhance.Brightness(imagen)
    darken_image = enhancer.enhance(factor)

    return darken_image