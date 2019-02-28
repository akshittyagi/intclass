#!/usr/bin/env python
# coding: utf-8

# In[8]:


import json
import numpy as np
from pprint import pprint

with open('test.json') as f:
    testraw = json.load(f)
    
with open('train.json') as f:
    trainraw = json.load(f)

pprint(trainraw)


# In[12]:


#get only intent and text
test = {}
test1 = []
train = {}
train1 = []
dev = {}
dev1 = []
temp1 = testraw['rasa_nlu_data']['common_examples']
for exp in temp1:
    onesamp = {}
    onesamp['intent'] = exp['intent']
    onesamp['text'] = exp['text']
    test1.append(onesamp)
test['body'] = test1

temp2 = trainraw['rasa_nlu_data']['common_examples']

for exp in temp2:
    onesamp = {}
    onesamp['intent'] = exp['intent']
    onesamp['text'] = exp['text']
    train1.append(onesamp)
#pprint(train)


# In[15]:


import random

random.shuffle(train1)

dev_data = train1[:500]
train_data = train1[500:]
print(len(train_data), len(dev_data))

train['body'] = train_data
dev['body'] = dev_data
pprint(dev)


# In[16]:


with open('test.json', 'w') as outfile:
    json.dump(test, outfile)
    
with open('train.json', 'w') as outfile:
    json.dump(train, outfile)
    
with open('dev.json', 'w') as outfile:
    json.dump(dev, outfile)


# In[17]:


from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())

print(X.shape)


# In[19]:


print(len(test['body']))


# In[ ]:




