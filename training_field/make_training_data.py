from image_generator import place_images
from os import listdir, path
from random import choice, choices, randint

backgrounds_dir = path.join("real_samples", "backgrounds")
droplets_dir = path.join("real_samples", "droplets")
backgrounds_paths = [path.join(backgrounds_dir, f) for f in listdir(backgrounds_dir) if path.isfile(path.join(backgrounds_dir, f))]
droplet_paths = [path.join(droplets_dir, f) for f in listdir(droplets_dir) if path.isfile(path.join(droplets_dir, f))]

outputs_dirs = []

train_images_dir = path.join("datasets", "images", "train")
train_labels_dir = path.join("datasets", "labels", "train")
outputs_dirs.append((train_images_dir, train_labels_dir))

val_images_dir = path.join("datasets", "images", "val")
val_labels_dir = path.join("datasets", "labels", "val")
outputs_dirs.append((val_images_dir, val_labels_dir))

for images_dir, labels_dir in outputs_dirs:
        for j in range(640):
            droplet_sample_size = randint(1,len(droplet_paths)-1)
            droplets_paths = choices(droplet_paths, k=droplet_sample_size)
            background_path = choice(backgrounds_paths)

            place_images(
                big_image_path = background_path,
                small_images_path = droplets_paths,
                num_small_images = randint(0,400),
                output_image_path = path.join(images_dir, str(j) +".jpg"),
                output_labels_path = path.join(labels_dir, str(j) + ".txt")
            )