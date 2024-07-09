# Introduction
This repository implements machine learning to detect droplets on images. After the detection is performed, the dimentions on these images are calculated. The dimentions can be shown in real time with a connected webcam or just for a single image.

# Main functionalities
There are three programms in this repository that a normal user would like to run: **main.py**, **see_ellipses.py** and **see_distributions.py**.

## main.py
This programm starts the webcam of your device and takes pictures periodicly. With each picture, the programm runs a prediction of the droplets in it and then collects the data of the dimentions of these. The data of the dimentions is then constantly used to update the visual interface of the density distribution of width, height and area. 

There are two modes for **main.py**: Graph mode and Terminal mode. If Graph mode is chosen, then the data will be shown as three bar graphs. Each of them with a curve for the density distribution of the data, made with the mean and the standard deviation, asuming a normal distribution. If terminal is chosen, then only the numbers will be shown and the bar graphs and the curve will be omitted. The terminal mode is a good option if the user wants to save comptutation power.

Both modes will have three possible interactions: **Exit**, togle **Pause** and **Forget**. When **Exit** is chosen the programm will be stopped and closed. When togle **Pause** is chosen it will pause the addition of the data of new webcam captures until togle **Pause** is chosen again. If **Forget** is chosen, the current data will be substituted with the data of one empty image.

The average speed of **main.py** in terminal mode is of 30 fps, while on graph mode is of 20 fps. Graph mode also stops for a few moments sometimes. If someone were to continue and improve this repository I think improving the graph mode should be a priority.

## see_ellipses.py
This programm predicts the droplets in an image, then it draws an ellipse over each of them, creating a new image that shows what was detected. The programm gives two option when it is started. To do this for only one image, which has to be specified in **PARAMETERS.py** or to do this for all images in **imgs\real_imgs**. The resulting image will be saved in **imgs\results\name_of_image** allong with a gif that shows the result compared to previous versions of the detection model.

## see_distributions.py
This programm also does a prediction over an image. After doing the prediction, it will print on the terminal the data of the distribution from width, height and area of the droplets that were detected. Allong with opening a new window that shows three bar graphs with quantities of the detected sizes and a curve for the density distribution of the droplets, asuming a normal distribution.

### PARAMETERS.py
Change the values in **PARAMETERS.py** in order to change: the pixel ratio, the unit of meassurement, the weight used in the model, the image that is analized in **see_ellipses.py** and **see_distributions**, among other things.

# Some Results

### snapshot 45
<img src="imgs/readme_imgs/snapshot_45_10.jpg" alt="snapshot 45" style="width: 600px; height: auto;">
<img src="imgs/readme_imgs/45_graphs.png" alt="graph 45" style="width:700px; height: auto;">

### snapshot 22
<img src="imgs/readme_imgs/snapshot_22_10.jpg" alt="snapshot 22" style="width: 600px; height: auto;">
<img src="imgs/readme_imgs/22_graphs.png" alt="graph 22" style="width: 700px; height: auto;">

### snaphsot 45 + snapshot 22
<img src="imgs/readme_imgs/45+22_graphs.png" alt="graph 22 + 45" style="width: 700px; height: auto;">
</br>

# MATHS AND PROCESSING

## Discarded droplets
In the first image seen above, one might think that all of the droplets at the borders are wrongfully not being detected. However this is not the case. The model does detect them, but they are not counted. The reason for this is that most of the droplets at the borders do not appear whole. Since their actual size is unknown, their inclusion would drift the mean and standard deviation of the dimentions away from the real values. 

The following image is the same from above, but it does count the droplets in the borders. The problem can inmediatly be seen.

<img src="imgs/readme_imgs/45_border.jpg" alt="45 with borders" style="width: 600px; height: auto;">

## Area of droplets
The area of droplets it's calculated asumming droplets have the form of an ellipse, whichs' axes are paralel to the x and y axes from the main picture.

To illustrate, in green the real area that is calculated, in yellow the area that is missed by te calculation and in red the error area that is calculated.

![Droplet error area illustration](imgs/readme_imgs/area_illustration.png)

The calculation of the volume is more imprecise, as it asumes an spherical geometry. The diameter of this shphere is given by the mean of the height and the width from the bounding box of the droplet. 

## How data is added and an explanation of batches

The standard deviation of the parameters, such as width, height and area is calculated with the incremental formula of the standard deviation:

![Incremental stdd formula](imgs/readme_imgs/incremental_stdd.png)

This formula allows to "add" the standard deviation of two or more images together, without having to store all of the individual dimentions of droplets. Basicly, after calculating the standard deviation once, the values from which it was calculated can be forgotten, since they won't be necessary when we want to calculate the new standard deviation from adding another set of droplets.

All data colected from the model's prediction is stored in an instance of `ImageData`. This instances can be added with each other easily. By doing `image_data_combined = image_data1 + image_data2` you will get a new instance of `ImageData`. This instance will have as attributes: the new mean given by considering the two images, the new standard deviation, the new data for making the graphs and the new total ammount of droplets.

The way this data is stored isn't straight forward. To allow to forget images after a given time, the data is stored in batches of a given size. These batches can not be more numerous than the defined maximum ammount of batches. 

An example of batches: If we have added 20 images into an instance of `ImageData`, which has a batch size of 10 and a maximum ammount of batches of 2. The instance would be storing the maximum ammount of data permited. So, the current mean value of the width is all of the widths from all of the images added up, divided by the ammount of droplets from the 20 images. But if we add another image, the instance would forget the first batch, since its maximum ammount of batches has been surpassed. Consequently, the instance would "forget" the first ten images and would consider only the previous last ten images plus the newly added image.

Currently the batches are configured to be of size 60 and to not surpass a quantity of 5. If we assume a refresh rate of 60 frames per second, this would mean storing the data of the previous five seconds. Therefore, when 5 seconds are surpassed, the first second of data is forgotten.

## Pixel ratio and units

Because of the nature of the pictures, dimentions of droplets are obtained in pixels. To make data more adaptable, two parameters were added to **PARAMETERS.py**: **PIXEL_RATIO** and **UNIT**. This allows to get the sizes of droplets in different units of meassurement. By default the **PIXEL_RATIO** and the **UNIT** are set to `1.0` and `"px"` respectively. Here is an example of what would change if we configured **PARAMETERS.py** for a camera that takes pictures where 1 pixel is 0.5 milimeters.

### Image of the following graphs
<img src="imgs/readme_imgs/snapshot_59_10.jpg" alt="history gif" style="width: 600px; height: auto;">

```py
PIXEL_RATIO = 1
UNIT = "px"
```
<img src="imgs/readme_imgs/pixel_ratio_1.png" alt="history gif" style="width: 600px; height: auto;">

```py
PIXEL_RATIO = 0.5
UNIT = "mm"
```
<img src="imgs/readme_imgs/pixel_ratio_2.png" alt="history gif" style="width: 600px; height: auto;">

What remains the same bewteen the two images are the graphs themselfs. This is: the bars, the curve and the shape. What differs is the values (except for the quantities). Since **PIXEL_RATIO** halfed, all of the numbers in the Width and Height graphs also halfed. In the case of the area, the values decreased to a fourth (0.5*0.5) of what they were.

# THE TRAINING FIELD
The weights are created in the training_field directory with the **train.py** file. This training is configured to use a nvidia graphics card with the NVIDIA CUDA toolkit. By doing this, the processing occurs in the GPU. This greatly improves the speed in which the training is done, but requires to download nvidia CUDA, nvidia CUDNN and to get a compatible version of PYTorch.

What the training does is: make the model predict the droplets on hundreds of images and see how correct the prediction is. This is done by checking the labels (coordinates of the droplets). The model then learns from this and keeps what worked and discards what didn't. Finally the model gives this "what worked and what didn't" in form of a weight (a .pt file), which is what we use in **main.py**.

The training is performed on "artificially" generated images and labels. Which are created using the **make_training_data.py** file.

## Training Images
This images are created by pasting hand-made cutouts of real droplets on random places of real background photos. These real samples are stored in the **real_samples** directory in the **training_field** and the Training Images are stored in **datasets**. On this directory the labels of the pasted droplets are also stored on the labels directory. 

New droplets are added to **real_samples** frequently. These are often edge cases that the model didn't recognize, so by adding them we can train the model again, so that it can recognize these "weird" droplets next time.

However, adding new droplets is not the only technique used to improve the detection rate. When placed, droplets are randomly: rotated, mirrored, streched, given transparency and darkened. These transformations have proven to be very usefull, and have made the general detection error much lower.

Here is a visualization of the detection model's history of improvements. The upleft number is the version of the weight that was used for the prediction. At version five the darkening transformation was implemented. Later others such as stretching and transparency were implemented as well.

<img src="imgs/readme_imgs/history.gif" alt="history gif" style="width: 600px; height: auto;">

The droplets in generated images will not overlap with eachother, except for a 4 pixel margin. This is made to better simulate the droplet proximity of real photos. This value was previously higher, allowing for more overlap, which caused the bigger circles seen in the gif for weights 7 and 8.

The droplets and backgrounds used are choosen randomly for every training image.

The background image is cropped to 640 pixels x 640 pixels when generating training images. The original size of 1024*704 became to heavy for the training, making it spend twice the memory and taking twice the time. 640 x 640 was a good medium point to train faster. So, to be clear. This change of size DOES NOT impact the precission of the final product, but it DOES increase the speed of the training.

# Requirements
The libraries needed to run main.py are **ultralytics**, **tabulate**. These can be automaticly installed by running **packages.py**. 

(New libraries are added normally, so these requirements could be outdated)
