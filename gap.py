#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2019年9月7日

@author: Administrator
'''
from __future__ import print_function

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import os
import codecs
from sklearn.metrics import accuracy_score
import numpy
from sklearn.model_selection import train_test_split
import pkuseg
import datetime
import pickle
from functools import partial
import math

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Embedding, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.utils import to_categorical
from numpy import argmax

from keras.callbacks import EarlyStopping
# data_dir = 'data/Artistes_et_Phalanges-David_Campion'  # data directory containing input.txt
# save_dir = 'save'  # directory to store models
rnn_size = 128  # size of RNN
batch_size = 30  # minibatch size
seq_length = 15  # sequence length
num_epochs = 8  # number of epochs
learning_rate = 0.001  # learning rate
sequences_step = 1  # step to create sequences

ngram_range = 1
max_features = 20000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 5

train_percent = 0.8
STATE = 1234


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


for filename in os.listdir(u"./cases"):
    if filename != 'execution.txt':
        continue
    data = []
    data_labels = []
    
    if filename.endswith(".txt"):
        token = filename.replace(".txt", "")
        
#         with codecs.open("./cases/" + filename, 'r', encoding='utf-8') as f:
#             for text in f:
#                 xy = text.split('|')
#                 if len(xy) > 1:
#                     data.append(xy[1])
#                     data_labels.append(xy[0])
#         print(len(data), len(data_labels))
        
        if token in ["civil", "criminal"]:
            with open("./numpy/" + token + "csr.pkl", "rb") as f:
                s = f.read()
                features_nd = pickle.loads(s)
                
                print("3 numpy pickle:", len(s))
        else:
            features_nd = numpy.load("./numpy/" + token + ".npz")["nd"]
            print("3:" + token + ".npz")
            
        with open("./tags/" + token + "-y.pkl", "rb") as f:
            s = f.read()
            data_labels = pickle.loads(s)
            print("3-1 tags pickle:", len(s))
        
        lbe = LabelEncoder()
        data_labels = lbe.fit_transform(data_labels).tolist()
        data_labels = to_categorical(data_labels, dtype=int)
        print("3-2 label encode:", len(data_labels), data_labels.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(features_nd,
                                                            data_labels,
                                                            train_size=train_percent,
                                                            random_state=STATE)
        print("4:split", train_percent)
        
        before_training = datetime.datetime.now()
        # build the model: a single GAP
        print('5:Build GAP model.')
        model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        model.add(Embedding(features_nd.shape[1],
                            embedding_dims,
                            input_length=features_nd.shape[1]))
        
        # we add a GlobalAveragePooling1D, which will average the embeddings
        # of all words in the document
        model.add(GlobalAveragePooling1D())
        
        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(data_labels.shape[1], activation='softmax'))  
              
        # adam optimizer
        optimizer = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        earlystop = [EarlyStopping(monitor='loss', min_delta=1e-3, patience=3, verbose=1, mode='min')]
        # fit the model
        model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=earlystop)
        after_training = datetime.datetime.now()
        
        with open("./model/" + token + "gap.pkl", "wb") as f:
            s = pickle.dumps(model)
            f.write(s)
            print("5-2 pickle:", len(s))
          
        train_pred = model.predict(X_train)
        print(token, '@train-accuracy-score', model.evaluate(X_train, y_train, batch_size=batch_size))
        print("5:training time(sec):", str((after_training - before_training).total_seconds()))
        print("5.1:decode expect", lbe.inverse_transform(numpy.argmax(y_train, axis=1)), ",actual:", lbe.inverse_transform(argmax(train_pred, axis=1))) 
        
        test_pred = model.predict(X_test)
        print(token, '@test-score', model.evaluate(X_test, y_test, batch_size=batch_size))
        print("6:test")
        print("6.1:decode expect", lbe.inverse_transform(argmax(y_test, axis=1)), ",actual:", lbe.inverse_transform(argmax(test_pred, axis=1)))
