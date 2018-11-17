# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:11:34 2018

@author: Austin Hsu
"""

import os
import pandas as pd
import numpy as np

data_test = pd.read_csv('test.csv').values
x_test = data_test[:,1].reshape((data_test.shape[0],1)).tolist()

for i in range(len(x_test)):
    x_test[i] = np.array([int(j) for j in x_test[i][0].split(' ')]).reshape((48,48,1)).tolist()

x_test = np.array(x_test)/255


from keras.models import Sequential, load_model

data_paths = os.listdir("saved_model/")
prediction = [0 for i in range(9)]

for i in range(9):
    model_path = data_paths[-i-1]
    model = Sequential()
    model = load_model("saved_model/"+model_path+"/"+"hw3_model.h5")
    prediction[i] = model.predict_classes(x_test)

final_vote = []
for i in range(prediction[0].shape[0]):
    index_vote = np.array([prediction[j][i] for j in range(9)])
    counts = np.bincount(index_vote)
    final_vote.append(np.argmax(counts))

with open('result.csv', 'w') as f:
    print('id,label', file = f)
    for i in range(len(final_vote)):
        print('%d,%d' % (i,final_vote[i]), file = f)


