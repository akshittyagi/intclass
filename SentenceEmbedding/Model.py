import numpy as np

from Embeddings import Embed
from Neural import SingleLayer

class SentenceEmbedder(object):

    def __init__(self, train_data, dev_data, embedding='glove', dim=300, min_count=1, epochs=10):
        self.tr = train_data
        self.dev = dev_data
        self.embedding = embedding
        self.dim = dim
        self.min_count = min_count
        self.epochs = epochs

    def organise_data(self):
        zipped_data_tr = zip(*self.tr)
        sentences_tr = list(zipped_data_tr[0])
        sentences_tr = [sentence.split() for sentence in sentences_tr]
        classes_tr = list(zipped_data_tr[1])
        classes_uniq = list(set(classes_tr))
        self.hashed_classes = {intent:idx for idx, intent in enumerate(classes_uniq)}
        classes_tr = [self.hashed_classes[intent] for intent in classes_tr]
        self.X = sentences_tr
        self.y = classes_tr

    def generate_embeddings(self):
        emb = Embed(self.X, embedding=self.embedding, dim=self.dim, min_count=self.min_count, epochs=self.epochs)
        emb.train()
        embeddings = []
        for sentence in self.X:
            curr_sentence = np.zeros(self.dim)
            for word in sentence:
                embed = emb.model[word]
                curr_sentence += embed
            curr_sentence /= len(sentence)
            embeddings.append(curr_sentence)
        return embeddings

    def train(self, train, dev):
        self.organise_data()
        X_embed = self.generate_embeddings()
        X_embed = np.array(X_embed)
        y = np.array(self.y)
    
    def test(self, test):
        pass