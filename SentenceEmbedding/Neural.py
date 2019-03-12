import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

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
