import numpy as np
import sys
sys.path.insert(0, "D:/OneDrive - hust.edu.cn/Study/LectureMaterial/CS690DS/NaiveBayesModel/intclass-master/SentenceEmbedding")
from Data import DataCleaner
from Data import data_loader
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
import csv

def getCorpusAndIntents(dataset):
    corpus = []
    intents = []
    for i in np.arange(len(dataset)):
        corpus.append(dataset[i][0])
        temp = dataset[i][1]
        intents.append(temp.rsplit(":")[1])
    return corpus, intents


# Data preparation for all the all the train, dev and test
data_tr, data_dev, data_test = data_loader("D:/OneDrive - hust.edu.cn/Study/LectureMaterial/CS690DS/DataSet/top-dataset-semantic-parsing-FB/")
tr_corpus, tr_intents = getCorpusAndIntents(data_tr)
dev_corpus, dev_intents = getCorpusAndIntents(data_dev)
test_corpus, test_intents = getCorpusAndIntents(data_test)

# encode all the categories into integer values
le = preprocessing.LabelEncoder()
le.fit(tr_intents)
tr_intents = le.transform(tr_intents)
dev_intents = le.transform(dev_intents)
test_intents = le.transform(test_intents)

# Transform raw text data into feature vectors using TF-IDF word level
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
tr_corpus_matrix = tfidf_vect.fit_transform(tr_corpus)
dev_corpus_matrix = tfidf_vect.transform(dev_corpus)
test_corpus_matrix = tfidf_vect.transform(test_corpus)

# Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(tr_corpus_matrix, tr_intents)
dev_intents_prd = clf.predict(dev_corpus_matrix)
test_intents_prd = clf.predict(test_corpus_matrix)
# print(test_intents)
# print(test_intents_prd)

# Evaluation of The Classifier:
evaluation = precision_recall_fscore_support(test_intents, test_intents_prd, average=None, labels=np.arange(len(le.classes_)))
print(evaluation)

with open("result.csv", "w",  newline = "") as f:
  writer = csv.writer(f, delimiter=",")
  writer.writerows(zip(test_intents, test_intents_prd))
