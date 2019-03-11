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

    def organise_data(self, mode='train', test_data=None):
        if mode == 'train':
            zipped_data_tr = list(zip(*self.tr))
            sentences_tr = list(zipped_data_tr[0])
            sentences_tr = [sentence.split() for sentence in sentences_tr]
            classes_tr = list(zipped_data_tr[1])
            classes_uniq = list(set(classes_tr))
            self.hashed_classes = {intent: idx for idx, intent in enumerate(classes_uniq)}
            classes_tr = [self.hashed_classes[intent] for intent in classes_tr]
            self.X = sentences_tr
            self.y = classes_tr
        if mode == 'test':
            zipped_data_tst = list(zip(*test_data))
            sentences = list(zipped_data_tst[0])
            sentences = [sentence.split() for sentence in sentences]
            classes = [self.hashed_classes[intent] for intent in list(zipped_data_tst[1])]
            return sentences, classes

    def generate_embeddings(self, mode='train', test_data=None):
        if mode == 'train':
            emb = Embed(self.X, embedding=self.embedding, dim=self.dim, min_count=self.min_count, epochs=self.epochs, debug=self.debug)
            self.embedding_obj = emb
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
        if mode == 'test':
            X = []
            for sentence in test_data:
                curr_sentence = np.zeros(self.dim)
                for word in sentence:
                    if word in self.embedding_obj.model:
                        curr_sentence += self.embedding_obj.model[word]
                curr_sentence /= len(sentence)
                X.append(curr_sentence)
            X = np.array(X)
            return X

    def train(self, train, dev):
        self.organise_data()
        if self.debug:
            print("Data Organized")
        X_embed = self.generate_embeddings()
        if self.debug:
            print("Embeddings Generated")
        
        self.device = ""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if os.path.exists(os.path.join(os.getcwd(), 'av_sent_emb_glove.MODEL')):
            self.neural_model = torch.load(os.path.join(os.getcwd(), 'av_sent_emb_glove.MODEL'))
            self.neural_model.to(self.device)
            return

        X_embed = torch.from_numpy(np.array(X_embed)).double()
        y = torch.from_numpy(np.array(self.y)).double()

        train_data = torch.utils.data.TensorDataset(X_embed, y)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

        X_embed = Variable(X_embed).float()
        y = Variable(y).type(torch.LongTensor)

        single_layer = SingleLayer(self.dim, len(self.hashed_classes))
        
        single_layer.to(self.device)
        optimizer = optim.SGD(single_layer.parameters(), lr=self.learning_rate)
        for epoch in range(self.neural_epochs):
            if self.debug:
                print("At epoch: ", epoch + 1)
            for idx, x in enumerate(X_embed):
                if self.debug and idx%10000 == 0:
                    print ("At datapoint: ", idx)
                single_layer.train()
                y_idx = y[idx]
                x = x.to(self.device)
                y_idx = y_idx.to(self.device)
                scores = single_layer(x)
                scores = torch.reshape(scores, (1, -1))
                y_idx = y_idx.reshape(1)
                loss = F.cross_entropy(scores, y_idx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        torch.save(single_layer, os.path.join(os.getcwd(), 'av_sent_emb_glove.MODEL'))
        self.neural_model = single_layer

    def test(self, test):
        X, y = self.organise_data(mode='test', test_data=test)
        X = self.generate_embeddings(mode='test', test_data=X)
        
        X = torch.from_numpy(np.array(X)).double()
        y = torch.from_numpy(np.array(y)).double()
        
        train_data = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

        X = Variable(X).float()
        y = Variable(y).type(torch.LongTensor)
        pred = []
        for idx, x in enumerate(X):
            y_idx = y[idx]
            x = x.to(self.device)
            y_idx = y_idx.to(self.device)
            scores = self.neural_model(x)
            _, indices = torch.max(scores, 0)
            pred.append(indices.item())
        pred = np.array(pred)
        y = y.data.numpy()
        accuracy = np.sum(y == pred) * 1.0 / len(y)
        print("Accuracy: ", accuracy)
