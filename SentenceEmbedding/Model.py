import os
import time
import random

import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.variable import Variable

from Embeddings import Embed
from Neural import SingleLayer, ThreeLayer, StackedLSTM, ThreeLayerBN, StackedLSTMBN
from utils import get_branchy_exit_weights, get_entropy_thresholds, accuracy
from sklearn.metrics import f1_score


class SentenceEmbedder(object):

    def __init__(self, train_data, dev_data, embedding='glove', dim=300, min_count=1, epochs=10, batch_size=64):
        self.tr = train_data
        self.dev = dev_data
        self.embedding = embedding
        self.dim = dim
        self.min_count = min_count
        self.epochs = epochs
        self.batch_size = batch_size
        self.neural_epochs = 5
        self.learning_rate = 1e-3
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
            sentences = []
            classes = []
            for idx in range(len(zipped_data_tst[0])):
                if zipped_data_tst[1][idx] not in self.hashed_classes:
                    continue
                sentences.append(zipped_data_tst[0][idx].split())
                classes.append(self.hashed_classes[zipped_data_tst[1][idx]])
            return sentences, classes

    def generate_embeddings(self, mode='train', net='fcn', test_data=None):
        if mode == 'train':
            emb = Embed(self.X, embedding=self.embedding, dim=self.dim, min_count=self.min_count, epochs=self.epochs, debug=self.debug)
            self.embedding_obj = emb
            emb.train()
            embeddings = []
            for sentence in self.X:
                if net == 'fcn':
                    curr_sentence = np.zeros(self.dim)
                    for word in sentence:
                        if word in self.embedding_obj.model:
                            embed = self.embedding_obj.model[word]
                            curr_sentence += embed
                    curr_sentence /= len(sentence)
                if net == 'rnn':
                    curr_sentence = np.zeros([20, self.dim])
                    for idx, word in enumerate(sentence):
                        if idx < 20:
                            if word in self.embedding_obj.model:
                                embed = self.embedding_obj.model[word]
                                curr_sentence[idx] = embed
                        else:
                            break

                embeddings.append(curr_sentence)
            return embeddings
        if mode == 'test':
            X = []
            for sentence in test_data:
                if net == 'fcn':
                    curr_sentence = np.zeros(self.dim)
                    for word in sentence:
                        if word in self.embedding_obj.model:
                            curr_sentence += self.embedding_obj.model[word]
                    curr_sentence /= len(sentence)
                    X.append(curr_sentence)
                if net == 'rnn':
                    curr_sentence = np.zeros([20, self.dim])
                    for idx, word in enumerate(sentence):
                        if idx < 20:
                            if word in self.embedding_obj.model:
                                embed = self.embedding_obj.model[word]
                                curr_sentence[idx] = embed

                    X.append(curr_sentence)
            X = np.array(X)
            return X

    def train(self, train, dev, model_type='recurrent'):
        if model_type == 'recurrent' or model_type == 'recurrent_bn':
            network = 'rnn'
        else:
            network = 'fcn'
        self.model_type = model_type
        self.organise_data()
        if self.debug:
            print("Data Organized")
        X_embed = self.generate_embeddings(net=network)
        if self.debug:
            print("Embeddings Generated")

        self.device = ""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        X_embed = torch.from_numpy(np.array(X_embed)).double()
        y = torch.from_numpy(np.array(self.y)).double()

        train_data = torch.utils.data.TensorDataset(X_embed, y)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

        X_embed = Variable(X_embed).float()
        y = Variable(y).type(torch.LongTensor)

        if model_type == 'feed_forward':
            model = ThreeLayer(self.dim, len(self.hashed_classes))
        elif model_type == 'recurrent':
            model = StackedLSTM(output_dim=len(self.hashed_classes), embedding_dim=self.dim)
        elif model_type == 'feed_forward_bn':
            entropies = []
            exit_weights = get_branchy_exit_weights(num=3, span=[0, 1])
            model = ThreeLayerBN(input_dim=self.dim, output_dim=len(self.hashed_classes), dimensions=[100, 75, 50], init_exit_weights=exit_weights)
        else:
            entropies = []
            exit_weights = get_branchy_exit_weights(num=3, span=[0, 1])
            model = StackedLSTMBN(output_dim=len(self.hashed_classes), embedding_dim=self.dim, init_exit_weights=exit_weights)

        if os.path.exists(os.path.join(os.getcwd(), 'bn_av_sent_emb_3_layer_glove.MODEL')):
            self.neural_model = torch.load(os.path.join(os.getcwd(), 'bn_av_sent_emb_3_layer_glove.MODEL'))
            self.neural_model.to(self.device)
            return

        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)

        num_examples = len(X_embed)
        batches = [(start, start + self.batch_size) for start in range(0, num_examples, self.batch_size)]

        for epoch in range(self.epochs):
            if self.debug:
                print("At epoch: ", epoch)
            av_loss = 0.
            start_time = time.time()
            random.shuffle(batches)
            for idx, (start, end) in enumerate(batches):
                batch = X_embed[start:end]
                if idx % 100 == 0:
                    print("At batch: ", idx)
                y_idx = y[start:end]
                batch = batch.to(self.device)
                y_idx = y_idx.to(self.device)
                if model_type == 'feed_forward_bn' or model_type == 'recurrent_bn':
                    scores, entropy = model(batch, y_idx)
                    entropies.append(entropy)
                    loss = scores
                else:
                    scores = model(batch)
                    # scores = torch.reshape(scores, (1, -1))
                    # y_idx = y_idx.reshape(1)
                    scores = scores.squeeze()
                    loss = F.cross_entropy(scores, y_idx)
                av_loss += loss.data.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print("Loss: ", av_loss / len(batch))
            print("Time:", time.time() - start_time)
        if model_type == 'feed_forward_bn' or model_type == 'recurrent_bn':
            percent_data = 0.3
            model.set_entropy_thresholds(get_entropy_thresholds(entropies, percent_data))
        torch.save(model, os.path.join(os.getcwd(), 'bn_av_sent_emb_3_layer_glove.MODEL'))
        self.neural_model = model

    def test(self, test, model_type='recurrent'):
        if model_type == 'recurrent' or model_type == 'recurrent_bn':
            network = 'rnn'
        else:
            network = 'fcn'
        if model_type.split("_")[-1] == 'bn':
            batching = 'off'
        else:
            batching = 'on'
        self.device = ""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.neural_model = torch.load(os.path.join(os.getcwd(), 'bn_av_sent_emb_3_layer_glove.MODEL'))
        self.neural_model.to(self.device)
        X, y = self.organise_data(mode='test', test_data=test)
        X = self.generate_embeddings(mode='test', test_data=X, net=network)

        X = torch.from_numpy(np.array(X)).double()
        y = torch.from_numpy(np.array(y)).double()

        train_data = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

        X = Variable(X).float()
        y = Variable(y).type(torch.LongTensor)
        pred = []
        num_examples = len(X)
        
        if batching == 'on':
            batches = [(start, start + self.batch_size) for start in range(0, num_examples, self.batch_size)]
            for idx, (start, end) in enumerate(batches):
                batch = X[start:end]
                y_idx = y[start:end]
                batch = batch.to(self.device)
                y_idx = y_idx.to(self.device)
                if self.model_type == 'feed_forward':
                    scores = self.neural_model(batch)
                    _, indices = torch.max(scores, 1)
                    indices = indices.cpu()
                    indices = np.array(indices)
                    pred.extend(indices)
        else:
            for idx, x in enumerate(X):
                x = x.to(self.device)
                y_idx = y[idx]
                y_idx = y_idx.to(self.device)
                if self.model_type == 'feed_forward_bn':  
                    scores = self.neural_model.forward_test(x)
                    _, indices = torch.max(scores, 0)
                    pred.append(indices.data.item())
                else:
                    if self.model_type == 'recurrent_bn':
                        scores = self.neural_model.forward_test(batch)
                    else:
                        scores = self.neural_model(batch)
                    indices = torch.argmax(scores, 2)
                    indices = torch.transpose(indices, 0, 1).squeeze()
                    indices = indices.cpu()
                    indices = np.array(indices)
                    pred.extend(indices)
        pred = np.array(pred)
        y = y.data.numpy()
        acc = accuracy(y, pred)
        print("Accuracy: ", acc)
        print("F1 macro: ", f1_score(y, pred, average='macro'))
        print("F1 micro: ", f1_score(y, pred, average='micro'))
