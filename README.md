# Behavioral-Cloning
Teach a convolutional neural network (NVIDIA architecture) how to drive using the Udacity self-driving car simulator 

* Create an Anaconda environment using **conda env create -f environment.yml --name car_environment** within the repo
* Activate the Anaconda environment using **source activate car_environment**

* model.py --> script used to create and train the model
* drive.py --> script to drive the car by Udacity
* model.h5 --> a trained Keras model
* video.py --> script used to create a video from the autonomous driving pictures

Save a video of the autonomous agent --> python drive.py model.h5 run1 (run1 being the directory where the images get saved)

Make an mp4 video of the autonomous driving --> python video.py run1 (--fps 48 for 48 FPS)

Starter code provided by [Udacity](https://github.com/udacity/CarND-Behavioral-Cloning-P3) 

