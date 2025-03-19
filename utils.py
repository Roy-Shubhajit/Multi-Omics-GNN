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

from typing import Optional, Tuple

import torch
from torch import Tensor


def dense_mincut_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    temp: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
    #print(f"x shape = {x.shape}, s shape = {s.shape}, adj shape = {adj.shape}")
    #print(f"out shape = {out.shape}, out_adj shappe = {out_adj.shape}")
    # MinCut regularization.
    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum('ijk->ij', adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    EPS = 1e-15

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)
    return out, out_adj, mincut_loss, ortho_loss


def _rank3_trace(x: Tensor) -> Tensor:
    return torch.einsum('ijj->i', x)


def _rank3_diag(x: Tensor) -> Tensor:
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1))

    return out

def create_graph(adj, data):
    graphs = []
    edge_index, edge_attr = to_edge_index(adj)
    for sample in tqdm(data.values):
        x = torch.tensor(sample[2:-1].astype(float), dtype=torch.float).half()
        y = torch.tensor([0 if sample[0]=='N' else 1], dtype=torch.long)
        graphs.append(Data(x=x.unsqueeze(1), y=y, edge_index=edge_index, edge_attr=edge_attr))
    return graphs

def load_data(args):
    if args.load_saved:
        print('Loading saved data')
        if args.keep_edge_weights:
            graph_data_train = torch.load('data/graph_data_train_with_edge_weights.pt')
            graph_data_test = torch.load('data/graph_data_test_with_edge_weights.pt')
            graph_data_val = torch.load('data/graph_data_val_with_edge_weights.pt')
        else:
            graph_data_train = torch.load('data/graph_data_train.pt')
            graph_data_test = torch.load('data/graph_data_test.pt')
            graph_data_val = torch.load('data/graph_data_val.pt')
        train_loader = DataLoader(graph_data_train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(graph_data_test, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(graph_data_val, batch_size=args.batch_size, shuffle=False)
        return train_loader, test_loader, val_loader
    else:
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
        
        if args.keep_edge_weights:
            train_data_adj[train_data_adj < args.adj_threshold] = 0
            test_data_adj[test_data_adj < args.adj_threshold] = 0
            val_data_adj[val_data_adj < args.adj_threshold] = 0
        else:
            train_data_adj = train_data_adj > args.adj_threshold
            test_data_adj = test_data_adj > args.adj_threshold
            val_data_adj = val_data_adj > args.adj_threshold
        
        train_data_adj_coo = sp.coo_matrix(train_data_adj)
        test_data_adj_coo = sp.coo_matrix(test_data_adj)
        val_data_adj_coo = sp.coo_matrix(val_data_adj)
        
        train_data_adj_torch = torch.sparse_coo_tensor(train_data_adj_coo.nonzero(), train_data_adj_coo.data, train_data_adj_coo.shape)
        test_data_adj_torch = torch.sparse_coo_tensor(test_data_adj_coo.nonzero(), test_data_adj_coo.data, test_data_adj_coo.shape)
        val_data_adj_torch = torch.sparse_coo_tensor(val_data_adj_coo.nonzero(), val_data_adj_coo.data, val_data_adj_coo.shape)
        
        graph_data_train = create_graph(train_data_adj_torch, train_data)
        graph_data_test = create_graph(test_data_adj_torch, test_data)
        graph_data_val = create_graph(val_data_adj_torch, val_data)
        
        if args.keep_edge_weights:
            torch.save(graph_data_train, 'data/graph_data_train_with_edge_weights.pt')
            torch.save(graph_data_test, 'data/graph_data_test_with_edge_weights.pt')
            torch.save(graph_data_val, 'data/graph_data_val_with_edge_weights.pt')
        else:
            torch.save(graph_data_train, 'data/graph_data_train.pt')
            torch.save(graph_data_test, 'data/graph_data_test.pt')
            torch.save(graph_data_val, 'data/graph_data_val.pt')
        
        train_loader = DataLoader(graph_data_train, batch_size=args.batch_size, shuffle=True, pin_memory=False)
        test_loader = DataLoader(graph_data_test, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(graph_data_val, batch_size=args.batch_size, shuffle=False)
        
        return train_loader, test_loader, val_loader
    
    

    
      