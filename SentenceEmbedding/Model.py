import os

import numpy as np
import torch
import torch.utils.data
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
        self.debug = True

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
        emb = Embed(self.X, embedding=self.embedding, dim=self.dim, min_count=self.min_count, epochs=self.epochs, debug=self.debug)
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
        if self.debug:
            print("Data Organized")
        X_embed = self.generate_embeddings()
        if self.debug:
            print("Embeddings Generated")
        
        # X_embed = torch.from_numpy(np.array(X_embed)).double()
        # y = torch.from_numpy(np.array(self.y)).double()
        
        X_embed = torch.from_numpy(np.array(X_embed)).double()
        y = torch.from_numpy(np.array(self.y)).double()

        train_data = torch.utils.data.TensorDataset(X_embed, y)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

        X_embed = Variable(X_embed).float()
        y = Variable(y).type(torch.LongTensor)

        single_layer = SingleLayer(self.dim, len(self.hashed_classes))
        device = ""
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        single_layer.to(device)
        optimizer = optim.SGD(single_layer.parameters(), lr=self.learning_rate)
        for epoch in range(self.neural_epochs):
            if self.debug:
                print("At epoch: ", epoch + 1)
            for idx, x in enumerate(X_embed):
                if self.debug and idx%10000 == 0:
                    print ("At datapoint: ", idx)
                single_layer.train()
                y_idx = y[idx]
                x = x.to(device)
                y_idx = y_idx.to(device)
                scores = single_layer(x)
                scores = torch.reshape(scores, (1, -1))
                y_idx = y_idx.reshape(1)
                loss = F.cross_entropy(scores, y_idx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        torch.save(single_layer, os.path.join(os.getcwd(), 'av_sent_emb_glove.MODEL'))

    def test(self, test):
        pass
