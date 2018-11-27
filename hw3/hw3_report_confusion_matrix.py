# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:01:20 2018

@author: Austin Hsu
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix#for confusion matrix
import matplotlib.pyplot as plt
from keras.utils import np_utils
import os
import itertools

data_train = pd.read_csv('train.csv').values
x_train = data_train[:,1].reshape((data_train.shape[0],1)).tolist()
y_train = data_train[:,0]

for i in range(len(x_train)):
    x_train[i] = np.array([int(j) for j in x_train[i][0].split(' ')]).reshape((48,48,1)).tolist()

x_train = np.array(x_train)/255.0

"""
validation_splitter = 0.1
data_len = x_train.shape[0]
data_arange = np.arange(data_len)
np.random.shuffle(data_arange)
x_val = np.array([x_train[i] for i in data_arange[:int(validation_splitter*data_len)]])
x_train = np.array([x_train[i] for i in data_arange[int(validation_splitter*data_len):]])
"""

y_train = np_utils.to_categorical(y_train)
"""
y_val = np.array([y_train[i] for i in data_arange[:int(validation_splitter*data_len)]])
y_train = np.array([y_train[i] for i in data_arange[int(validation_splitter*data_len):]])
"""

from keras.models import Sequential, load_model

data_paths = os.listdir("saved_model/")
prediction = [0 for i in range(9)]

for i in range(9):
    model_path = data_paths[-i-1]
    model = Sequential()
    model = load_model("saved_model/"+model_path+"/"+"hw3_model.h5")
    prediction[i] = model.predict_classes(x_train)

final_vote = []
for i in range(prediction[0].shape[0]):
    index_vote = np.array([prediction[j][i] for j in range(9)])
    counts = np.bincount(index_vote)
    final_vote.append(np.argmax(counts))
    
conf_matrix = confusion_matrix(y_train, np.array(final_vote))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

plt.figure()
plot_confusion_matrix(conf_matrix, classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
plt.show()