#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2019年5月17日

@author: Administrator
'''
from sklearn.feature_extraction.text import CountVectorizer
import os
import codecs
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy
from sklearn.model_selection import train_test_split
import pkuseg
import datetime
import pickle
from functools import partial

seg = pkuseg.pkuseg(model_name='web')


def cut(text, aseg):
    return aseg.cut(text)


cutanalyzer = partial(cut, aseg=seg)

train_percent = 0.8
STATE = 1234

before_model = datetime.datetime.now()
        
for filename in os.listdir(u"./cases"):
    if filename == 'execution.txt':
        continue
    
    data = []
    data_labels = []
    
    if filename.endswith(".txt"):
        token = filename.replace(".txt", "")
        
        with codecs.open("./cases/" + filename, 'r', encoding='utf-8') as f:
            for text in f:
                xy = text.split('|')
                if len(xy) > 1:
                    data.append(xy[1])
                    data_labels.append(xy[0])
        
        print(len(data), len(data_labels))
        
        features_nd = numpy.load("./numpy/" + token + ".npz")["nd"]
        print("3:" + token + ".npz")
        with open("./tags/" + token + "-y.pkl", "rb") as f:
            s = f.read()
            data_labels = pickle.loads(s)
            print("3-1 tags pickle:", len(s))
            
        X_train, X_test, y_train, y_test = train_test_split(features_nd,
                                                            data_labels,
                                                            train_size=train_percent,
                                                            random_state=STATE)
        print("4:split", train_percent)
        
        before_training = datetime.datetime.now()
        mlp_model = MLPClassifier()
        mlp_model = mlp_model.fit(X=X_train, y=y_train)
        after_training = datetime.datetime.now()
          
        with open("./model/" + token + "mlp.pkl", "wb") as f:
            s = pickle.dumps(mlp_model)
            f.write(s)
            print("5-2 pickle:", len(s))
           
        train_pred = mlp_model.predict(X_train)
        print(token, '@train-accuracy-score', accuracy_score(y_train, train_pred))
        print("5:training time(sec):", str((after_training - before_training).total_seconds()))
          
        y_pred = mlp_model.predict(X_test)
        print(token, '@test-score', accuracy_score(y_test, y_pred))
        print("6:test")
        
        continue
    else:
        continue

after_model = datetime.datetime.now()
print("ALL-DONE", str((after_model - before_model).total_seconds()))
