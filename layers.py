import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphNeuralNet(Module):

    def __init__(self, batch_size, in_features, out_features):
        super(GraphNeuralNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size

        weight1_eye = torch.FloatTensor(torch.eye(in_features, out_features))
        weight1_eye = weight1_eye.reshape((1, in_features, out_features))
        weight1_eye = weight1_eye.repeat(batch_size, 1, 1)
        self.weight1 = Parameter(weight1_eye)
        self.weight2 = Parameter(torch.zeros(batch_size, in_features, out_features))

    def forward(self, input, adj):
        v1 = torch.bmm(input, self.weight1)
        v2 = torch.bmm(torch.bmm(adj, input), self.weight2)
        output = v1 + v2
        return output