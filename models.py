import torch.nn as nn
import torch.nn.functional as F
from layers import GraphNeuralNet
import torch
from utils import doublerelu

class GNN(nn.Module):

    def __init__(self, batch_size, nfeat, ndim):
        super(GNN, self).__init__()
        self.gc1 = GraphNeuralNet(batch_size, ndim, ndim)

    def forward(self, x, adj):
        x = doublerelu(self.gc1(x, adj))
        x = x/x.sum(axis=2).unsqueeze(2) #normalize st sum = 1
        return x
