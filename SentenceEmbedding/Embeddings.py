import os
import pickle as pkl
from gensim.models import KeyedVectors, Word2Vec

class Embed(object):

    def __init__(self, sentences, embedding='glove', dim=300, min_count=1, epochs=10):
        '''
            train
        '''
        self.sentences = sentences
        self.embedding = embedding
        self.dimension = dim
        self.min_count = min_count
        self.epochs = epochs

    def train(self):
        pretrained_model_path = ""
        pretrained_model = []
        if self.embedding == 'glove':
            pretrained_model_path = '../DataSets/glove.6B.300d.txt.word2vec'
            pretrained_model = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=False)
        model = Word2Vec(size=self.dimension, min_count=self.min_count)
        model.build_vocab(sentences)
        dataset_size = model.corpus_count
        model.build_vocab([pretrained_model.vocab.keys()], update=True)
        model.intersect_word2vec_format(pretrained_model_path, binary=False, lockf=1.0)
        model.train(sentences, total_examples=dataset_size, epochs=self.epochs)
        model_name = pretrained_model_path.split("/")[2] + "_EMBED.pkl"
        self.model = model
        pkl.dump(self.model, open(model_name, 'wb'))
        print("Model saved: ", model_name)
    
