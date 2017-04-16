# **Behavioral Cloning**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./mse_plot.png " Model Mean Squared Error (MSE) by Epoch"

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

The model contains a Keras lamba layer to normalize the channels in the image (model.py line 73) and a Keras Cropping layer to crop the top half and of the camera images as well as the car hood in the bottom. (model.py lines 72 and 73)

#### 2. Attempts to reduce overfitting in the model

Overfitting was manually monitored by plotting the Mean Squared Error of the model for both the training data and the validation data versus the epoch. The plot made for the traning of the model.h5 file is shown below:

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

The final model was unadultarated nvidia architecture (model.py lines 71-83)

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

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
