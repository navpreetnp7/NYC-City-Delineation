import numpy as np
import torch
import networkx as nx
import pandas as pd

def load_data():

    data = pd.read_csv('data/LEHD_nyc.csv', delimiter=',')
    data = data.rename(columns={'flow': 'weight'})
    G = nx.from_pandas_edgelist(data, 'origin', 'destination', 'weight')
    return np.array([nx.adjacency_matrix(G).todense()], dtype=float)


def normalize(adj):

    adj = torch.FloatTensor(adj)
    adj_id = torch.FloatTensor(torch.eye(adj.shape[1]))
    adj_id = adj_id.reshape((1, adj.shape[1], adj.shape[1]))
    adj_id = adj_id.repeat(adj.shape[0], 1, 1)
    adj = adj + adj_id
    rowsum = torch.FloatTensor(adj.sum(2))
    degree_mat_inv_sqrt = torch.diag_embed(torch.float_power(rowsum,-0.5), dim1=-2, dim2=-1).float()
    adj_norm = torch.bmm(torch.transpose(torch.bmm(adj,degree_mat_inv_sqrt),1,2),degree_mat_inv_sqrt)

    return adj_norm


def doublerelu(x):
    return torch.clamp(x, 0, 1)