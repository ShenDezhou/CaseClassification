#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2019年5月17日

@author: Administrator
'''
from sklearn.feature_extraction.text import CountVectorizer
import os
import codecs
from sklearn.linear_model import LogisticRegression
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
Dense=False

before_model = datetime.datetime.now()
        
for filename in os.listdir(u"./cases"):
    if filename != 'civil.txt':
        continue
    data = []
    data_labels = []
    log_model = LogisticRegression()
    if filename.endswith(".txt"):
        token = filename.replace(".txt", "")
        with codecs.open("./cases/" + filename, 'r', encoding='utf-8') as f:
            for text in f:
                xy = text.split('|')
                if len(xy) > 1:
                    data.append(xy[1])
                    data_labels.append(xy[0])
        
        print(len(data), len(data_labels))
        
        vectorizer = CountVectorizer(
            analyzer=cutanalyzer,
            lowercase=False,
        )
        features = vectorizer.fit_transform(
            data
        )
          
        with open("./model/" + token + "countvectorizer.pkl", "wb") as f:
            s = pickle.dumps(vectorizer)
            f.write(s)
            print("2:vectorizer", len(s))

        if Dense:
            features_nd = features.toarray()
            numpy.savez_compressed("./numpy/" + token + ".npz", nd=features_nd)
            print("3:" + token + ".npz")
        else:
            features_nd = features.tocsr()
            with open("./numpy/" + token + "csr.pkl", "wb") as f:
                s = pickle.dumps(features_nd)
                f.write(s)
                print("3 numpy:", len(s))
            
        
        with open("./tags/" + token + "-y.pkl", "wb") as f:
            s = pickle.dumps(data_labels)
            f.write(s)
            print("3-1 tags pickle:", len(s))
            
        X_train, X_test, y_train, y_test = train_test_split(features_nd,
                                                            data_labels,
                                                            train_size=train_percent,
                                                            random_state=STATE)
        print("4:split", train_percent)
        
        before_training = datetime.datetime.now()
        log_model = LogisticRegression(solver='newton-cg', max_iter=100, random_state=STATE,
                                     multi_class='multinomial').fit(X=X_train, y=y_train)
            
        after_training = datetime.datetime.now()
        
        with open("./model/" + token + "logistic.pkl", "wb") as f:
            s = pickle.dumps(log_model)
            f.write(s)
            print("5-1 pickle:", len(s))
         
        train_pred = log_model.predict(X_train)
        print(token, '@train-accuracy-score', accuracy_score(y_train, train_pred))
        
        print("5:training time(sec):", str((after_training - before_training).total_seconds()))
        
        y_pred = log_model.predict(X_test)
        print(token, '@test-score', accuracy_score(y_test, y_pred))
        print("6:test")
        
        continue
    else:
        continue

after_model = datetime.datetime.now()
print("ALL-DONE", str((after_model - before_model).total_seconds()))
