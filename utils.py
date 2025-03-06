import csv
import pandas as pd
import numpy as np
import anndata as ad
from tqdm import tqdm
import PyWGCNA as pwc
from torch_geometric.data import Data
from torch_geometric.utils import to_edge_index
from torch_geometric.loader import DataLoader
import scipy.sparse as sp
import torch

def create_graph(adj, data):
    graphs = []
    edge_index, edge_attr = to_edge_index(adj)
    for sample in tqdm(data.values):
        x = torch.tensor(sample[2:-1].astype(float), dtype=torch.float)
        y = torch.tensor([0 if sample[0]=='N' else 1], dtype=torch.long)
        graphs.append(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr))
    return graphs

def load_data(args):
    with open('data/data.csv') as f:
        reader = csv.reader(f)
        #remove white space 
        data = [row for row in reader]
        data = [[x.strip() for x in row] for row in data]
        data_pd = pd.DataFrame(data[1:], columns=data[0])
    data_pd['Sample_ID'] = range(1, len(data_pd) + 1)
    #split data_pd into train, test and validation
    train_data = data_pd.sample(frac=args.train_ratio, random_state=0)
    test_data = data_pd.drop(train_data.index)
    val_data = train_data.sample(frac=args.val_ratio, random_state=0)
    train_data = train_data.drop(val_data.index)
    
    
    train_data_X = train_data.drop(columns=['Sample_ID', 'Label'])
    test_data_X = test_data.drop(columns=['Sample_ID', 'Label'])
    val_data_X = val_data.drop(columns=['Sample_ID', 'Label'])
    
    train_data_X.to_csv('data/train_data_X.csv', index=False)
    test_data_X.to_csv('data/test_data_X.csv', index=False)
    val_data_X.to_csv('data/val_data_X.csv', index=False)
    train_data_X_ad = ad.io.read_csv('data/train_data_X.csv')
    test_data_X_ad = ad.io.read_csv('data/test_data_X.csv')
    val_data_X_ad = ad.io.read_csv('data/val_data_X.csv')
    train_data_adj = pwc.WGCNA.adjacency(train_data_X_ad.to_df())
    test_data_adj = pwc.WGCNA.adjacency(test_data_X_ad.to_df())
    val_data_adj = pwc.WGCNA.adjacency(val_data_X_ad.to_df())
    
    train_data_adj[train_data_adj < args.adj_threshold] = 0
    test_data_adj[test_data_adj < args.adj_threshold] = 0
    val_data_adj[val_data_adj < args.adj_threshold] = 0
    
    train_data_adj_coo = sp.coo_matrix(train_data_adj)
    test_data_adj_coo = sp.coo_matrix(test_data_adj)
    val_data_adj_coo = sp.coo_matrix(val_data_adj)
    
    train_data_adj_torch = torch.sparse_coo_tensor(train_data_adj_coo.nonzero(), train_data_adj_coo.data, train_data_adj_coo.shape)
    test_data_adj_torch = torch.sparse_coo_tensor(test_data_adj_coo.nonzero(), test_data_adj_coo.data, test_data_adj_coo.shape)
    val_data_adj_torch = torch.sparse_coo_tensor(val_data_adj_coo.nonzero(), val_data_adj_coo.data, val_data_adj_coo.shape)
    
    graph_data_train = create_graph(train_data_adj_torch, train_data)
    graph_data_test = create_graph(test_data_adj_torch, test_data)
    graph_data_val = create_graph(val_data_adj_torch, val_data)
    
    train_loader = DataLoader(graph_data_train, batch_size=1, shuffle=True)
    test_loader = DataLoader(graph_data_test, batch_size=1, shuffle=False)
    val_loader = DataLoader(graph_data_val, batch_size=1, shuffle=False)
    
    return train_loader, test_loader, val_loader
    
    

    
      