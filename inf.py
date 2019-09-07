#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2019年9月7日

@author: Administrator
'''
from sklearn.feature_extraction.text import CountVectorizer
import os
import codecs
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy
from sklearn.model_selection import train_test_split
import pkuseg
import datetime
import pickle
from functools import partial
from pathlib import Path

if __name__ == "__main__":
    token = 'execution'
    with codecs.open("./cases/" + token + ".txt", 'r', encoding='utf-8') as f:
        for text in f:
            xy = text.split('|')
            if len(xy) > 1:
                sample = xy[1]
                sample_labels = xy[0]
                break
        
    with open("./model/" + token + "countvectorizer.pkl", "rb") as f:
        s = f.read()
        print("1:vectorizer", len(s))
        vectorizer = pickle.loads(s)
        
    
    features = vectorizer.transform(
            sample
        )
    features_nd = features.toarray()
    
    with open("./model/" + token + "decisiontree.pkl", "rb") as f:
        s = f.read()
        cart_model = pickle.loads(s)
        print("2:decisiontree", len(s))
    
    sample_pred = cart_model.predict(features_nd)
    print("3:expected:", sample_labels, "actual:", sample_pred)
    
    print("ALL-DONE")
    
