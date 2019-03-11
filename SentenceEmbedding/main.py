import os

import sklearn

from Data import data_loader
from Model import SentenceEmbedder

if __name__ == "__main__":
    dataset_loc = "../DataSets/top-dataset-semantic-parsing/"
    data_train, data_dev, data_test = data_loader(dataset_loc, dataset='fb')
    sentence_embedder = SentenceEmbedder(train_data=data_train, dev_data=data_dev)
    sentence_embedder.train(data_train, data_dev)
    sentence_embedder.test(data_test)