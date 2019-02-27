import os
import multiprocessing
import pickle as pkl
from gensim.models import KeyedVectors, Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec

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
        import pdb; pdb.set_trace()
        if self.embedding == 'glove':
            pretrained_model_path = '../DataSets/glove.6B/glove.6B.300d.txt'
            glove2word2vec(glove_input_file=pretrained_model_path,word2vec_output_file='../DataSets/glove.6B/glove.6B.300d.w2v')
            pretrained_model = KeyedVectors.load_word2vec_format('../DataSets/glove.6B/glove.6B.300d.w2v', binary=False)
        model = Word2Vec(size=self.dimension, min_count=self.min_count, workers=multiprocessing.cpu_count())
        model.build_vocab(sentences)
        dataset_size = model.corpus_count
        model.build_vocab([pretrained_model.vocab.keys()], update=True)
        model.intersect_word2vec_format(pretrained_model_path, binary=False, lockf=1.0)
        model.train(sentences, total_examples=dataset_size, epochs=self.epochs)
        model_name = pretrained_model_path.split("/")[2] + "_EMBED.pkl"
        self.model = model
        pkl.dump(self.model, open(model_name, 'wb'))
        print("Model saved: ", model_name)
    
