import os
import time
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
import PyWGCNA as pwc
from torch_geometric.data import Data
from torch_geometric.utils import to_edge_index
import torch
from utils import *
from network import Classify_with_GAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--adj_threshold', type=float, default=0.3)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--reduction_factor', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    
    train_loader, test_loader, val_loader = load_data(args)
    args.in_dim = train_loader.dataset[0].x.size(1)
    args.n1 = train_loader.dataset[0].x.size(0)
    args.n2 = args.n1 // args.reduction_factor
    args.n3 = args.n2 // args.reduction_factor
    
    model = Classify_with_GAE(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    classifical_loss = torch.nn.CrossEntropyLoss()
    GAE_loss = torch.nn.MSELoss()
    best_val_acc = 0
    best_test_acc = 0
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    for epoch in tqdm(range(args.epochs)):
        model.train()
        loss_mincut = 0
        loss_ortho = 0
        pred_GAE = torch.tensor([]).to(device)
        orig_GAE = torch.tensor([]).to(device)
        pred_classification = torch.tensor([]).to(device)
        orig_classification = torch.tensor([]).to(device)
        for data in train_loader:
            optimizer.zero_grad()
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            y = data.y.to(device)
            class_prob, x_bar, mincut_losses, ortho_losses = model(x, edge_index, edge_attr)
            pred_GAE = torch.cat((pred_GAE, x_bar), dim=0)
            orig_GAE = torch.cat((orig_GAE, x), dim=0)
            pred_classification = torch.cat((pred_classification, class_prob), dim=0)
            orig_classification = torch.cat((orig_classification, y), dim=0)
            loss_mincut += sum(mincut_losses)
            loss_ortho += sum(ortho_losses)
            
        train_loss = GAE_loss(pred_GAE, orig_GAE) + classifical_loss(pred_classification, orig_classification) + loss_mincut + loss_ortho
        train_acc = (pred_classification.argmax(dim=1) == orig_classification).sum().item() / orig_classification.size(0)
        train_loss.backward()
        optimizer.step()
        
        model.eval()
        loss_mincut = 0
        loss_ortho = 0
        pred_GAE = torch.tensor([]).to(device)
        orig_GAE = torch.tensor([]).to(device)
        pred_classification = torch.tensor([]).to(device)
        orig_classification = torch.tensor([]).to(device)
        for data in val_loader:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            y = data.y.to(device)
            class_prob, x_bar, mincut_losses, ortho_losses = model(x, edge_index, edge_attr)
            pred_GAE = torch.cat((pred_GAE, x_bar), dim=0)
            orig_GAE = torch.cat((orig_GAE, x), dim=0)
            pred_classification = torch.cat((pred_classification, class_prob), dim=0)
            orig_classification = torch.cat((orig_classification, y), dim=0)
            loss_mincut += sum(mincut_losses)
            loss_ortho += sum(ortho_losses)
        
        val_loss = GAE_loss(pred_GAE, orig_GAE) + classifical_loss(pred_classification, orig_classification) + loss_mincut + loss_ortho
        val_acc = (pred_classification.argmax(dim=1) == orig_classification).sum().item() / orig_classification.size(0)
        
        loss_mincut = 0
        loss_ortho = 0
        pred_GAE = torch.tensor([]).to(device)
        orig_GAE = torch.tensor([]).to(device)
        pred_classification = torch.tensor([]).to(device)
        orig_classification = torch.tensor([]).to(device)
        for data in test_loader:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            y = data.y.to(device)
            class_prob, x_bar, mincut_losses, ortho_losses = model(x, edge_index, edge_attr)
            pred_GAE = torch.cat((pred_GAE, x_bar), dim=0)
            orig_GAE = torch.cat((orig_GAE, x), dim=0)
            pred_classification = torch.cat((pred_classification, class_prob), dim=0)
            orig_classification = torch.cat((orig_classification, y), dim=0)
            loss_mincut += sum(mincut_losses)
            loss_ortho += sum(ortho_losses)
            
        test_loss = GAE_loss(pred_GAE, orig_GAE) + classifical_loss(pred_classification, orig_classification) + loss_mincut + loss_ortho
        test_acc = (pred_classification.argmax(dim=1) == orig_classification).sum().item() / orig_classification.size(0)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss
            best_val_acc = val_acc
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pth'))
            print("Model saved")
        if epoch % 10:
            print(f"Epoch: {epoch}, Train Loss: {train_loss.item()}, Train Acc: {train_acc}, Val Loss: {val_loss.item()}, Val Acc: {val_acc}, Test Loss: {test_loss.item()}, Test Acc: {test_acc}")
    print(f"Best Val Loss: {best_val_loss.item()}, Best Val Acc: {best_val_acc}, Best Test Loss: {best_test_loss.item()}, Best Test Acc: {best_test_acc}")
                 
            
            
            
    
    
    
    
    

