# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 20:58:44 2018

@author: Austin Hsu
"""

import pickle

history = pickle.load(open("TrainingHistoryDict","rb"))

import matplotlib.pyplot as plt

plt.figure()
plt.plot(history['val_acc'])
plt.xlabel('epoch')
plt.ylabel('validation accuracy')
plt.show()