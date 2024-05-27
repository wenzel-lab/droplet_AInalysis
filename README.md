To process an image, write on the terminal "python main.py" while being on the directory's root. This will print the number of droplets detected, the mean of the area and the standard deviation of the area. 
To run this, it is necessary to install Yolov8. It can be installed with the following command: "pip install ultralytics".

If you want to see the visualization of the model's prediction, run "python just_predict.py".

In the training_field directory you can create new images with the "make_training_data.py" file.
In order to train the model, you need to run "train.py". However this was optimized for my pc and for that reason, it is very likely that it will not work on your device right away. Since, the final product will not do any training, I consider this to not be a problem.
