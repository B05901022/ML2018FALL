#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 20:02:25 2018

@author: austinhsu
"""

import numpy as np
import jieba
from gensim.models import word2vec
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Flatten, Embedding, CuDNNGRU, Dense, TimeDistributed, BatchNormalization, LeakyReLU, Dropout, LSTM, GRU, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.callbacks import History

jieba.load_userdict('dict.txt.big')

w2v_model = word2vec.Word2Vec.load("dcard_word2vec.model")

#cut the sentence
input_datas = [list(jieba.cut(i.split(',')[1])) for i in open('./train_x.csv', 'r').read().split('\n')[1:-1]]

#load and generate train data/label
label = to_categorical(np.load("dcard_labels.npy")).reshape(120000,1,2)#np.load("dcard_labels.npy").reshape(120000,1,1)#
#train_data = text_to_index(input_datas)


tokenizer = Tokenizer(num_words=len(w2v_model.wv.vocab))
tokenizer.fit_on_texts(input_datas)
train_data = tokenizer.texts_to_sequences(input_datas)

"""
#padding
padding_length = 200
train_data = pad_sequences(train_data, maxlen=padding_length)
#train_data = to_categorical(train_data)
#train_data = train_data.reshape(120000,1,200)
"""


#DNN Model
model = Sequential()

model.add(Dense(512, input_shape=(1,28829)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dropout(0.7))


model.add(Dense(128))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dropout(0.7))


model.add(Dense(128))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dropout(0.7))

model.add(Dense(64))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dropout(0.7))


model.add(Dense(32))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dropout(0.6))

model.add(Dense(2, activation='softmax'))

model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_train_data=np.zeros((120000,28829))
for i in range(120000):
    for j in train_data[i]:
        batch_train_data[i,j] += 1
batch_train_data=batch_train_data.reshape(120000,1,28829)
batch_label=label

history = model.fit(x=batch_train_data, y=batch_label, batch_size=500, epochs=10, validation_split=0.1)

model.save('hw4_model_bow.h5')

#np.save("mean_std.npy", np.array(mean,std))

import pickle
with open('./TrainingHistoryDict_bow.pkl', 'wb') as f:
    pickle.dump(history.history, f)

with open('tokenizer.pkl', 'wb') as t:
    pickle.dump(tokenizer, t, protocol=pickle.HIGHEST_PROTOCOL)
    

    