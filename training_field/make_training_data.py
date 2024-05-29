from image_generator import place_images
from os import listdir, path
from random import choice, choices, randint

backgrounds_dir = path.join("real_samples", "backgrounds")
droplets_dir = path.join("real_samples", "droplets")

outputs_dirs = []
train_images_dir = path.join("datasets", "my_data", "images", "train")
train_labels_dir = path.join("datasets", "my_data", "labels", "train")
outputs_dirs.append((train_images_dir, train_labels_dir))
val_images_dir = path.join("datasets", "my_data", "images", "val")
val_labels_dir = path.join("datasets", "my_data", "labels", "val")
outputs_dirs.append((val_images_dir, val_labels_dir))

for images_dir, labels_dir in outputs_dirs:
    backgrounds_paths = [path.join(backgrounds_dir, f) for f in listdir(backgrounds_dir) if path.isfile(path.join(backgrounds_dir, f))]

    droplet_images = [path.join(droplets_dir, f) for f in listdir(droplets_dir) if path.isfile(path.join(droplets_dir, f))]

    i = 0
    for background_path in backgrounds_paths:
        for j in range(i*40, i*40 + 40):
            droplet_sample_size = randint(1,len(droplet_images)-1)
            droplets_paths = choices(droplet_images, k=droplet_sample_size)
            place_images(
                big_image_path = background_path,
                small_images_path = droplets_paths,
                num_small_images = randint(0,350),
                output_image_path = path.join(images_dir, str(j) +".jpg"),
                output_labels_path = path.join(labels_dir, str(j) + ".txt")
            )
        i += 1