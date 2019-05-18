import os
import time
import random

import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss

from Embeddings import Embed
from Neural import SingleLayer, ThreeLayer, StackedLSTM, ThreeLayerBN, StackedLSTMBN, FourLayerBN, FiveLayerBN, SixLayerBN, BertEarlyExit
from utils import get_branchy_exit_weights, get_entropy_thresholds, accuracy
from sklearn.metrics import f1_score

from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from Data import create_bert_examples, bert_examples_to_features
from tqdm import tqdm, trange


class SentenceEmbedder(object):

    def __init__(self, train_data, dev_data, embedding='glove', dim=300, min_count=1, epochs=10, batch_size=64, debug=True):
        self.tr = train_data
        self.dev = dev_data
        self.embedding = embedding
        self.dim = dim
        self.min_count = min_count
        self.epochs = epochs
        self.batch_size = batch_size
        self.neural_epochs = 5
        self.learning_rate = 1e-5
        # self.learning_rate = 3e-4
        self.debug = debug

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

    def train(self, train, dev, clip=0, model_type='recurrent'):
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
            exit_weights = get_branchy_exit_weights(num=6, span=[0, 1])
            model = SixLayerBN(input_dim=self.dim, output_dim=len(self.hashed_classes), dimensions=[100, 75, 50, 40, 25, 20], init_exit_weights=exit_weights)
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
                if self.debug and idx % 100 == 0:
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

                if clip:
                    total_norm = 0.
                    clip_grad_norm_(model.parameters(), clip)
                    for p in model.parameters():
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    if self.debug and idx % 100 == 0:
                        print('gradient norm: ', total_norm)

                # total_norm = 0.
                # for p in model.parameters():
                #     param_norm = p.grad.data.norm(2)
                #     total_norm += param_norm.item() ** 2
                # total_norm = total_norm ** (1. / 2)
                # print('gradient norm: ', total_norm)

                optimizer.step()
                optimizer.zero_grad()

            if self.debug:
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
        exit_points = {}

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
                elif self.model_type == 'recurrent':
                    scores = self.neural_model(batch)
                    indices = torch.argmax(scores, 2)
                    indices = torch.transpose(indices, 0, 1).squeeze()
                indices = indices.cpu()
                indices = np.array(indices)
                pred.extend(indices)
        else:
            for idx, x in enumerate(X):
                x = x.to(self.device)
                y_idx = y[idx]
                y_idx = y_idx.to(self.device)
                if self.model_type == 'feed_forward_bn':
                    exit_, scores = self.neural_model.forward_test(x)
                    _, indices = torch.max(scores, 0)
                    pred.append(indices.data.item())
                elif self.model_type == 'recurrent_bn':
                    exit_, scores = self.neural_model.forward_test(x)
                    indices = torch.argmax(scores, 2).squeeze()
                    # indices = torch.transpose(indices, 0, 1).squeeze()
                    # indices = indices.cpu()
                    # indices = np.array(indices)
                    pred.append(indices.data.item())

                if exit_ in exit_points:
                    exit_points[exit_] += 1
                else:
                    exit_points[exit_] = 1

        pred = np.array(pred)
        y = y.data.numpy()
        acc = accuracy(y, pred)
        print(exit_points)
        print([v / len(y) for k, v in exit_points.items()])
        print("Accuracy: ", acc)
        f1_mac = f1_score(y, pred, average='macro')
        f1_mic = f1_score(y, pred, average='micro')
        print("F1 macro: ", f1_mac)
        print("F1 micro: ", f1_mic)
        return acc, f1_mac, f1_mic

    def train_bert(self, test_data):
        zipped_data_tr = list(zip(*self.tr))
        classes_tr = list(zipped_data_tr[1])
        classes_uniq = list(set(classes_tr))
        self.hashed_classes = {intent: idx for idx, intent in enumerate(classes_uniq)}

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        train_examples = create_bert_examples(self.tr, 'train')
        num_train_opt_steps = int(len(train_examples) / self.batch_size) * self.epochs
        num_labels = len(self.hashed_classes)
        # model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
        #                                                       cache_dir='./bert_cache/',
        #                                                       num_labels=num_labels)
        exit_weights = get_branchy_exit_weights(num=2, span=[0, 1])
        model = BertEarlyExit.from_pretrained('bert-base-uncased',
                                                cache_dir='./bert_cache/',
                                                num_labels=num_labels)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if self.device == torch.device('cuda'):
            self.n_gpu = torch.cuda.device_count()
        model.to(self.device)

        if self.n_gpu and self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        print("n_gpu: {}".format(self.n_gpu))

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.learning_rate,
                             warmup=0.1,
                             t_total=num_train_opt_steps)

        global_step = 0
        tr_steps = 0
        tr_loss = 0

        train_features = bert_examples_to_features(train_examples, classes_uniq, 75, tokenizer)

        print('Training...')
        print('Num examples: {}'.format(len(train_examples)))
        print('Batch size: {}'.format(self.batch_size))
        print('Num steps: {}'.format(num_train_opt_steps))

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        model.train()

        for i in trange(self.epochs, desc="Epoch"):
            tr_loss = 0
            num_tr_examples, tr_steps = 0, 0
            ave_loss = 0.
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, labels=label_ids)

                # loss_fn = CrossEntropyLoss()
                # # loss = loss_fn(logits.view(-1, num_labels), label_ids.view(-1))
                # loss_first = loss_fn(first_layer_logits.view(-1, num_labels), label_ids.view(-1))
                # loss_last = loss_fn(first_layer_logits.view(-1, num_labels), label_ids.view(-1))
                

                if self.n_gpu > 1:
                    loss = loss.mean()

                if step % 10 == 0:
                    print('epoch {}, step {} loss: {}'.format(i, step, loss))

                av_loss += loss.data.item()
                loss.backward()

                tr_loss += loss.item()
                num_tr_examples += input_ids.size(0)
                tr_steps += 1

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            print("Avg Epoch {} Loss: ".format(i), av_loss / len(batch))

        filtered_tst_data = []
        for ex in test_data:
            if ex[1] not in self.hashed_classes:
                continue
            filtered_tst_data.append((ex[0], ex[1]))
        
        eval_examples = create_bert_examples(filtered_tst_data, 'test')
        eval_features = bert_examples_to_features(eval_examples, classes_uniq, 75, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data) # use all data
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)

        model.eval()
        eval_loss = 0
        eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)

            loss_fn = CrossEntropyLoss()
            eval_loss += loss_fn(logits.view(-1, num_labels), label_ids.view(-1)).mean().item()
            eval_steps += 1
            if len(preds) == 0:
                preds = logits.detach().cpu().numpy()
            else:
                pred = logits.detach().cpu().numpy()
                preds = np.append(preds, pred, axis=0)
            
        eval_loss = eval_loss / eval_steps
        preds = np.argmax(preds, axis=1)

        acc = accuracy(preds, all_label_ids.numpy())
        print("Accuracy: ", acc)
        f1_mac = f1_score(all_label_ids.numpy(), preds, average='macro')
        f1_mic = f1_score(all_label_ids.numpy(), preds, average='micro')
        print("F1 macro: ", f1_mac)
        print("F1 micro: ", f1_mic)
        return acc, f1_mac, f1_mic