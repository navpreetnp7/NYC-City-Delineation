import numpy as np
import torch
import networkx as nx
import pandas as pd

def load_data():

    data = pd.read_csv('data/LEHD_nyc.csv', delimiter=',')
    data = data.rename(columns={'flow': 'weight'})
    G = nx.from_pandas_edgelist(data, 'origin', 'destination', 'weight',create_using=nx.DiGraph())
    adj_list = np.array([nx.adjacency_matrix(G).todense()], dtype=float)
    init_feat = np.array(data.groupby('origin')['initialFeat'].agg(['unique']))
    true_label = np.array(data.groupby('origin')['true_label'].agg(['unique']))
    init_feat = np.array(list(map(lambda x: x[0], init_feat))).reshape(-1, 1)
    true_label = np.array(list(map(lambda x: x[0][0], true_label))).reshape(-1, 1)
    return adj_list,init_feat,true_label


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