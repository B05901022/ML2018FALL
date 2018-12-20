#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 18:16:21 2018

@author: austinhsu
"""

import numpy as np
import jieba
from gensim.models import word2vec
from keras.preprocessing.text import Tokenizer
import pickle

jieba.load_userdict('dict.txt.big')

w2v_model = word2vec.Word2Vec.load("dcard_word2vec.model")

#cut the sentence
input_datas = [list(jieba.cut(i.split(',')[1])) for i in open('./test_x.csv', 'r').read().split('\n')[1:-1]]



with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
test_data = tokenizer.texts_to_sequences(input_datas)

batch_test_data=np.zeros((80000,28829))
for i in range(80000):
    for j in test_data[i]:
        batch_test_data[i,j] += 1
batch_test_data=batch_test_data.reshape(80000,1,28829)


from keras.models import Sequential, load_model

model = Sequential()
model = load_model('hw4_model_bow.h5')
prediction = model.predict_classes(batch_test_data)

with open("result_bow.csv", 'w') as f:
    print('id,label', file = f)
    for i in range(prediction.shape[0]):
        print('%d,%d' % (i,prediction[i][0]), file = f)