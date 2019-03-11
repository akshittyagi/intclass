import os
import multiprocessing
import pickle as pkl

from gensim.models import KeyedVectors, Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec

from utils import EpochLogger, MonitorLossLogger

class Embed(object):

    def __init__(self, sentences, embedding='glove', dim=300, min_count=1, epochs=50, debug=False):
        '''
            train
        '''
        self.sentences = sentences
        self.embedding = embedding
        self.dimension = dim
        self.min_count = min_count
        self.epochs = epochs
        self.debug = debug

    def train(self):
        pretrained_model_path = ""
        pretrained_model = []
        if self.embedding == 'glove':
            pretrained_model_path = '../DataSets/glove.6B/glove.6B.300d.txt'
            model_name = pretrained_model_path.split("/")[2] + "_EMBED.pkl"
            if os.path.exists(model_name):
                self.model = pkl.load(open(model_name, 'rb'))
                return
            _ = glove2word2vec(glove_input_file=pretrained_model_path,word2vec_output_file='../DataSets/glove.6B/glove.6B.300d.w2v')
            if self.debug:
                print("...... Loading Pretrained GloVe")
            pretrained_model = KeyedVectors.load_word2vec_format('../DataSets/glove.6B/glove.6B.300d.w2v', binary=False)
        epoch_logger = EpochLogger()
        monitorloss_logger = MonitorLossLogger()
        model = Word2Vec(size=self.dimension, min_count=self.min_count, workers=multiprocessing.cpu_count(), callbacks=[epoch_logger, monitorloss_logger])
        model.build_vocab(self.sentences)
        dataset_size = model.corpus_count
        model.build_vocab([list(pretrained_model.vocab.keys())], update=True)
        model.intersect_word2vec_format('../DataSets/glove.6B/glove.6B.300d.w2v', binary=False, lockf=1.0)
        print("...... Fine tunigng GloVe for the given dataset")
        model.train(self.sentences, total_examples=dataset_size, epochs=self.epochs)
        self.model = model
        pkl.dump(self.model, open(model_name, 'wb'))
        print("Model saved: ", model_name)
