# Different functions that are used frequently, and so are needed in many files
from os.path import splitext, exists, join

def get_available_filename(filename):
    base, extension = splitext(filename)
    counter = 1
    new_filename = base + "_1" + extension
    
    # Generate a unique filename
    while exists(join(new_filename)):
        new_filename = f"{base}_{counter}{extension}"
        counter += 1

    return new_filename
