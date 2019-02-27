#!/usr/bin/env python
# coding: utf-8

# In[83]:


import json
import numpy as np
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[88]:


def readJson(loc = ''):
    
    with open(loc+'atistrain.json') as f:
        traina = json.load(f)
        
    with open(loc+'atistest.json') as f:
        testa = json.load(f)
    
    with open(loc+'atisdev.json') as f:
        deva = json.load(f)
        
    return traina['body'],testa['body'],deva['body']


#given a dictionary of text and intent, return the text in the list for tfidf
def getCorp(dataset):
    result = []
    for sample in dataset:
        result.append(sample['text'])
    return result

#get the correct labels
def getLabel(dataset):
    result = []
    for exp in dataset:
        result.append(exp['intent'])
        
    return result

def getInputX(traint, testt, devt):
    vectorizer = TfidfVectorizer()
    trainX = vectorizer.fit_transform(traint)
    testX = vectorizer.transform(testt)
    devX = vectorizer.transform(devt)

    return trainX, testX, devX

def getClassLabels(testy):
    classlabs = list(set(testy.copy()))
    return classlabs


# In[89]:


loca = '../DataSets/ATIS/'
trainX, testX, devX = getInputX(getCorp(readJson(loca)[0]), getCorp(readJson(loca)[1]), getCorp(readJson(loca)[2]))
trainy = getLabel(readJson()[0])
testy = getLabel(readJson()[1])
devy = getLabel(readJson()[2])

clf = MultinomialNB()
clf.fit(trainX, trainy)

y_predtest = clf.predict(testX)
y_preddev = clf.predict(devX)

#get accuracy on the test set
(testy == y_predtest).sum()/len(testd)

#for label in classlabs:
    #print(label, f1_score(testy, y_predtest, label, average='macro'))
    
#for label in classlabs:
    #print(label, precision_score(testy, y_predtest, label, average='macro'))
    
#for label in classlabs:
    #print(label, recall_score(testy, y_predtest, label, average='macro'))


# In[ ]:




