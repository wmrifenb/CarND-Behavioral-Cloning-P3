# **Behavioral Cloning**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./mse_loss.png " Model Mean Squared Error (MSE) by Epoch"
[image2]: ./examples/youtube_link.png "Course 2 video"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used nVidia's CNN architecture as described in their paper [here](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) .

The model contains a Keras lamba layer to normalize the channels in the image (model.py line 73) and a Keras Cropping layer to crop the top half and of the camera images as well as the car hood in the bottom. (model.py lines 72)

#### 2. Attempts to reduce overfitting in the model

Overfitting was manually monitored by plotting the Mean Squared Error Loss of the model for both the training data and the validation data versus the epoch. The plot made during the traning of the model.h5 file is shown below:

![alt text][image1]

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 85).

#### 4. Appropriate training data

Training data used was a combination of the data provided by Udacity of the first race track in [data.zip](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) and data I recorded using the simulator on the second race track.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Selection of the model architecture was to chose what nVidia had already developed for behovioral cloning in their paper. I also multiplied the amount of available training data by horizontally flipping the images (model.py lines 55-59) and using the images recorded by all cameras using an appropriate steering angle correction (model.py lines 46-53)

At first the nVidia model architecture had difficulty not driving into the water on the bend after the bridge. My first attempt to correct this was to add more training data. I recorded myself driving the deadly curve several times and recorded to image data into folders titled 'data_no_swimming_plz' and 'data_plz_no_swimming' (model.py lines 15 and 16) This data did little to help. I recorded data of myself driving the second race track to help generalize the network and the autonomous driving seemed to improve more but still didnt prevent driving into the water. I played with Dropout in between the Dense layers but that had little improvement.

What I eventually arrived upon is that increasing the number of epochs to 6 was all that was needed to get the nvida architecture working.

#### 2. Final Model Architecture

The final model was the unadulturated nvidia architecture (model.py lines 71-83)

```python
# NVIDIA Architecture
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

#### 3. Creation of the Training Set & Training Process

To generalize the model, I recorded one lap on the second track using center lane driving. Below is a link to a video of the entire course at 120 FPS

[![alt text][image2]](https://youtu.be/ICDDawfclNU)

This data combined with the data provided by Udacity's first track runs proved sufficient for training.

To augment the data, I also flipped images and angles and included the left and right camera images as well. I corrected the steering anlge for the left camera image by adding correction value and did the same for the right camera images by subtracting the correction value. I chose 0.2 as a starting point for steering correction and ran training several times before arriving at 0.25 as a good steering correction according to how well the model autonomously drove the car. Having a powerful nvidia GPU at my disposal enabled me to quickly do this.

I randomly shuffled the data set and put 20% of the data into a validation set. I used a generator to handle the massive amounts of image data in batches of 32 image samples each to mitigate memory issues during training (model.py lines 31-63)

Using six epochs, I trained the model that would succesfully drive one lap around the course:

[Click here for video.mp4](https://github.com/wmrifenb/CarND-Behavioral-Cloning-P3/blob/master/video.mp4)
