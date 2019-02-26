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
    
    