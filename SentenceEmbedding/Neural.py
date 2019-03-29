import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SingleLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(SingleLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        nn.init.kaiming_normal_(self.fc1.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x


class ThreeLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ThreeLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, output_dim)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class ThreeLayerBN(nn.Module):

    def __init__(self, input_dim, output_dim, dimensions, init_exit_weights):
        super(ThreeLayerBN, self).__init__()
        self.fc1 = nn.Linear(input_dim, dimensions[0])
        self.exit_1 = nn.Linear(dimensions[0], output_dim)
        self.scale_weight_1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(dimensions[0], dimensions[1])
        self.exit_2 = nn.Linear(dimensions[1], output_dim)
        self.scale_weight_2 = nn.Linear(1, 1)
        self.fc3 = nn.Linear(dimensions[1], dimensions[2])
        self.exit_3 = nn.Linear(dimensions[2], output_dim)
        self.scale_weight_3 = nn.Linear(1, 1)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.exit_1.weight)
        nn.init.kaiming_normal_(self.exit_2.weight)
        nn.init.kaiming_normal_(self.exit_3.weight)
        self.scale_weight_1.weight.data.fill_(init_exit_weights[0])
        self.scale_weight_2.weight.data.fill_(init_exit_weights[1])
        self.scale_weight_3.weight.data.fill_(init_exit_weights[2])
        self.scale_weight_1.bias.data.fill_(0)
        self.scale_weight_2.bias.data.fill_(0)
        self.scale_weight_3.bias.data.fill_(0)
        self.entropy_thresholds = [] * 3

    def forward(self, x, y):
        y = y.reshape(1)
        x = F.relu(self.fc1(x))
        exit_1 = self.exit_1(x)
        sm_1 = F.softmax(exit_1)
        neg_entropy_1 = torch.sum(sm_1 * torch.log(sm_1))
        loss_1 = self.scale_weight_1(F.cross_entropy(torch.reshape(exit_1, (1, -1)), y).reshape(1, 1))
        x = F.relu(self.fc2(x))
        exit_2 = self.exit_2(x)
        sm_2 = F.softmax(exit_2)
        neg_entropy_2 = torch.sum(sm_2 * torch.log(sm_2))
        loss_2 = self.scale_weight_2(F.cross_entropy(torch.reshape(exit_2, (1, -1)), y).reshape(1, 1))
        x = F.relu(self.fc3(x))
        exit_3 = self.exit_3(x)
        sm_3 = F.softmax(exit_3)
        neg_entropy_3 = torch.sum(sm_3 * torch.log(sm_3))
        loss_3 = self.scale_weight_3(F.cross_entropy(torch.reshape(exit_3, (1, -1)), y).reshape(1, 1))
        return (loss_1 + loss_2 + loss_3) / 3, [neg_entropy_1, neg_entropy_2, neg_entropy_3]    

    def set_entropy_thresholds(self, thresholds):
        self.entropy_thresholds = thresholds

    def forward_test(self, x):
        x = F.relu(self.fc1(x))
        exit_1 = self.exit_1(x)
        sm_1 = F.softmax(exit_1, dim=0)
        neg_entropy_1 = torch.sum(sm_1 * torch.log(sm_1))
        if neg_entropy_1 < self.entropy_thresholds[0]:
            return exit_1
        x = F.relu(self.fc2(x))
        exit_2 = self.exit_2(x)
        sm_2 = F.softmax(exit_2, dim=0)
        neg_entropy_2 = torch.sum(sm_2 * torch.log(sm_2))
        if neg_entropy_2 < self.entropy_thresholds[1]:
            return exit_2
        x = F.relu(self.fc3(x))
        exit_3 = self.exit_3(x)
        return exit_3

class StackedLSTM(nn.Module):

    def __init__(self, output_dim, hidden_dim=1200, embedding_dim=300, num_layers=3):
        super(StackedLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.inp = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.rnns = [nn.LSTM(self.hidden_dim,
                             self.hidden_dim, batch_first=True)
                     for i in range(self.num_layers)]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

        nn.init.kaiming_normal_(self.inp.weight)
        nn.init.kaiming_normal_(self.out.weight)

    def forward(self, x):
        inp = self.inp(x)
        lstm_out = inp.unsqueeze(0)

        for i, layer in enumerate(self.rnns):
            lstm_out, (h, c) = layer(lstm_out)

        logits = self.out(lstm_out)

        return logits
