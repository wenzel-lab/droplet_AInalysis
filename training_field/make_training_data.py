from image_generator import place_images
from os import listdir, path
from random import choice, choices, randint

backgrounds_dir = path.join("images", "backgrounds")
droplets_dir = path.join("images", "droplets")
result_images_dir = path.join("datasets", "my_data", "images", "train")
result_labels_dir = path.join("datasets", "my_data", "labels", "train")


background_images = [path.join(backgrounds_dir, f) for f in listdir(backgrounds_dir) if path.isfile(path.join(backgrounds_dir, f))]
backgrounds_paths = choices(background_images,k=8)

droplet_images = [path.join(droplets_dir, f) for f in listdir(droplets_dir) if path.isfile(path.join(droplets_dir, f))]

i = 0
for background_path in backgrounds_paths:
    for j in range(i*21, i*21 + 21):
        droplet_sample_size = randint(1,len(droplet_images)-1)
        droplets_paths = choices(droplet_images, k=droplet_sample_size)
        place_images(
            big_image_path = background_path,
            small_images_path = droplets_paths,
            num_small_images = randint(0,500),
            output_image_path = path.join(result_images_dir, str(j) +".jpg"),
            output_labels_path = path.join(result_labels_dir, str(j) + ".txt")
        )
    i += 1