# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 15:35:19 2018

@author: Austin Hsu
"""

import pandas as pd
import numpy as np
from keras.utils import np_utils

data_train = pd.read_csv('train.csv').values
x_train = data_train[:,1].reshape((x_train.shape[0],1)).tolist()
y_train = data_train[:,0]

for i in range(len(x_train)):
    x_train[i] = [np.array(x_train[i][0].split(' ')).reshape((48,48)).tolist()]

x_train = np.array(x_train)

y_train = np_utils.to_categorical(y_train)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', input_shape = (1,48,48), activation = 'relu' ))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2), padding = 'same'))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2), padding = 'same'))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2), padding = 'same'))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(7, activation = 'softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy', 'accuracy'])

datagen = keras.preprocessing.image.ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rotation_range=20,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), steps_per_epoch=len(x_train) / 32, epochs=200)

model.save('hw3_model.h5')