import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.layers import Cropping2D
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# This will allow us to pick which recordings we actually like
csvfilestoread = ['./data/driving_log.csv',
                  #                  './data_no_swimming_plz/driving_log.csv',
                  #                  './data_plz_no_swimming/driving_log.csv'
                  ]

samples = []
for csvfiletoread in csvfilestoread:
    with open(csvfiletoread) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[3] != 'steering':
                samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(img_samples, batch_size=32):
    num_samples = len(img_samples)
    while 1:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = img_samples[offset:offset + batch_size]

            images = []
            angles = []
            correction_angle = 0.25
            center, left, right = 0, 1, 2
            for camera_view in range(2):
                for batch_sample in batch_samples:
                    name = './data/IMG/' + batch_sample[camera_view].split('/')[-1]
                    sample_image = cv2.imread(name)

                    # Handle steering correction angle for given camera view
                    sample_angle = float(batch_sample[3])
                    if camera_view == left:
                        sample_angle += correction_angle
                    if camera_view == right:
                        sample_angle -= correction_angle
                    images.append(sample_image)
                    angles.append(sample_angle)

                    # Add flipped images and measurements
                    augmented_image = cv2.flip(sample_image, 1)
                    augmented_measurement = sample_angle * -1.0
                    images.append(augmented_image)
                    angles.append(augmented_measurement)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# NVIDIA Architecture
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
