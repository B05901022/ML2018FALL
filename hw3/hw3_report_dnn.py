# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 19:40:53 2018

@author: Austin Hsu
"""

import pandas as pd
import numpy as np
from keras.utils import np_utils

data_train = pd.read_csv('train.csv').values
x_train = data_train[:,1].reshape((data_train.shape[0],1)).tolist()
y_train = data_train[:,0]

for i in range(len(x_train)):
    x_train[i] = np.array([int(j) for j in x_train[i][0].split(' ')]).reshape((48,48,1)).tolist()

x_train = np.array(x_train)/255.0

validation_splitter = 0.1
data_len = x_train.shape[0]
data_arange = np.arange(data_len)
np.random.shuffle(data_arange)
x_val = np.array([x_train[i] for i in data_arange[:int(validation_splitter*data_len)]])
x_train = np.array([x_train[i] for i in data_arange[int(validation_splitter*data_len):]])

y_train = np_utils.to_categorical(y_train)
y_val = np.array([y_train[i] for i in data_arange[:int(validation_splitter*data_len)]])
y_train = np.array([y_train[i] for i in data_arange[int(validation_splitter*data_len):]])

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, LeakyReLU

model = Sequential()
model.add(Flatten(input_shape = (48,48,1)))
model.add(Dense(1024))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(1024))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(1024))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(256))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(256))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(64))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(64))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(32))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(32))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(32))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(16))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(16))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(16))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(8))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(8))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(8))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(7, activation = 'softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy', 'accuracy'])

datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2)

early_stopping=keras.callbacks.EarlyStopping(monitor='acc', patience=20, verbose=0, mode='auto')
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=100), steps_per_epoch=int(x_train.shape[0] / 100), validation_data=(x_val, y_val), epochs=200)#, callbacks=[early_stopping]

#model.save('hw3_model.h5')
import pickle
with open('./TrainingHistoryDictOfDNN', 'wb') as f:
    pickle.dump(history.history, f)