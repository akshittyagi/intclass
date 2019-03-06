import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.variable import Variable

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
        self.neural_epochs = epochs
        self.learning_rate = 1e-4

    def organise_data(self):
        zipped_data_tr = list(zip(*self.tr))
        sentences_tr = list(zipped_data_tr[0])
        sentences_tr = [sentence.split() for sentence in sentences_tr]
        classes_tr = list(zipped_data_tr[1])
        classes_uniq = list(set(classes_tr))
        self.hashed_classes = {intent: idx for idx, intent in enumerate(classes_uniq)}
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
        import pdb; pdb.set_trace()
        X_embed = self.generate_embeddings()
        X_embed = Variable(np.array(X_embed))
        y = Variable(np.array(self.y))
        single_layer = SingleLayer(self.dim, len(self.hashed_classes))
        device = ""
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        single_layer.to(device)
        optimizer = optim.SGD(single_layer.parameters(), lr=self.learning_rate)
        for epoch in range(self.neural_epochs):
            for idx, x in enumerate(X_embed):
                single_layer.train()
                y_idx = y[idx]
                x = x.to(device)
                y = y_idx.to(device)
                scores = single_layer(x)
                loss = F.cross_entropy(scores, y_idx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def test(self, test):
        pass
