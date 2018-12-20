#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 14:49:38 2018

@author: austinhsu
"""

import numpy as np
import jieba
from gensim.models import word2vec
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, CuDNNGRU, Dense, TimeDistributed, BatchNormalization, LeakyReLU, Dropout, LSTM, GRU, Bidirectional
import sys

jieba.load_userdict(sys.argv[4])
w2v_model = word2vec.Word2Vec.load("dcard_word2vec.model")

#embedding layer
embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
word2idx = {}

vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1
    
embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False
                            )


def text_to_index(corpus):
    new_corpus = []
    for doc in corpus:
        new_doc = []
        for word in doc:
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
        new_corpus.append(new_doc)
    return np.array(new_corpus)

#cut the sentence
input_datas = [list(jieba.cut(i.split(',')[1])) for i in open(sys.argv[1], 'r').read().split('\n')[1:-1]]

#load and generate train data/label
label = to_categorical(np.array([int(i.split(',')[1]) for i in open(sys.argv[2], 'r').read().split('\n')[1:-1]])).reshape(120000,1,2)#label = to_categorical(np.load("dcard_labels.npy")).reshape(120000,1,2)#np.load("dcard_labels.npy").reshape(120000,1,1)#
train_data = text_to_index(input_datas)


"""
#normalize?
mean = np.mean(train_data)
std = np.std(train_data)
train_data = (train_data-mean)/std
"""

#padding
padding_length = 200
train_data = pad_sequences(train_data, maxlen=padding_length)


#RNN model
model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(GRU(256, return_sequences=True)))
model.add(Bidirectional(GRU(256, return_sequences=True)))
#model.add(Dense(256, activation='relu'))
model.add(TimeDistributed(Dense(256, activation='relu')))
#model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
#model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.1))
model.add(Dense(2, activation = 'softmax'))

model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy', 'accuracy'])


history = model.fit(x=train_data, y=label, batch_size=500, epochs=5, validation_split=0.1)

model.save('hw4_model.h5')
#np.save("mean_std.npy", np.array(mean,std))

import pickle
with open('./TrainingHistoryDict', 'wb') as f:
    pickle.dump(history.history, f)


