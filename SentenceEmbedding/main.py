import os
import sklearn

from Data import data_loader
from Model import sentence_embedder

if __name__ == "__main__":
    dataset_loc = ""
    data_train, data_test = data_loader(dataset_loc)
    sentence_embedder.train(data_train)
    sentence_embedder.test(data_test)

