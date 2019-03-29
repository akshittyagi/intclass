import os

import sklearn

from Data import data_loader
from Model import SentenceEmbedder

if __name__ == "__main__":
    dataset_loc = "../DataSets/ATIS/"
    data_train, data_dev, data_test = data_loader(dataset_loc, dataset='atis')
    sentence_embedder = SentenceEmbedder(train_data=data_train, dev_data=data_dev, epochs=20)
    sentence_embedder.train(data_train, data_dev, model_type='feed_forward')
    sentence_embedder.test(data_test)