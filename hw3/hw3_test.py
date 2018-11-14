#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 09:55:02 2018

@author: bibiduck14
"""

import pandas as pd
import numpy as np

data_test = pd.read_csv('test.csv').values
x_test = data_test[:,1].reshape((data_test.shape[0],1)).tolist()

for i in range(len(x_test)):
    x_test[i] = np.array([int(j) for j in x_test[i][0].split(' ')]).reshape((48,48,1)).tolist()

x_test = np.array(x_test)/255

import keras
from keras.models import Sequential, load_model

model = Sequential()
model = load_model("hw3_model.h5")
prediction = model.predict_classes(x_test)

with open('result.csv', 'w') as f:
    print('id,label', file = f)
    for i in range(prediction.shape[0]):
        print('%d,%d' % (i,prediction[i]), file = f)