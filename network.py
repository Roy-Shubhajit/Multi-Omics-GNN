import torch
from torch_geometric.nn import GCNConv, dense_mincut_pool
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_edge_index

class GAE_Encoder(torch.nn.Module):
    def __init__(self, args, in_dim, hidden_dim):
        super(GAE_Encoder, self).__init__()
        self.conv1 = GCNConv(args.in_dim, args.hidden_dim)
        self.lt1 = torch.nn.Linear(args.hidden_dim, args.n2)
        self.conv2 = GCNConv(args.hidden_dim, args.hidden_dim)
        self.lt2 = torch.nn.Linear(args.hidden_dim, args.n3)
        self.conv3 = GCNConv(args.hidden_dim, args.hidden_dim)
        self.lt3 = torch.nn.Linear(args.hidden_dim, 1)
        
    def reset_parameter(self):
        self.conv1.reset_parameters()
        self.lt1.reset_parameters()
        self.conv2.reset_parameters()
        self.lt2.reset_parameters()
        self.conv3.reset_parameters()
        self.lt3.reset_parameters()
    
    def forward(self, x, edge_index1, edge_attr1):
        adj0 = to_dense_adj(edge_index=edge_index1, edge_attr=edge_attr1, max_num_nodes=x.size(0))
        x1 = F.relu(self.conv1(x, edge_index1, edge_attr1))
        s1 = F.relu(self.lt1(x1))
        x1_bar, adj1, mincut_loss1, ortho_loss1 = dense_mincut_pool(x1, adj1, s1)
        
        edge_index2, edge_attr2 = to_edge_index(adj1)
        x2 = F.relu(self.conv2(x1_bar, edge_index2, edge_attr2))
        s2 = F.relu(self.lt2(x2))
        x2_bar, adj2, mincut_loss2, ortho_loss2 = dense_mincut_pool(x2, adj1, s2)
        
        edge_index3, edge_attr3 = to_edge_index(adj2)
        x3 = F.relu(self.conv3(x2_bar, edge_index3, edge_attr3))
        s3 = F.relu(self.lt3(x3))
        x3_bar, adj3, mincut_loss3, ortho_loss3 = dense_mincut_pool(x3, adj2, s3)
        
        return x3_bar, adj3, (s1, s2, s3), (mincut_loss1, mincut_loss2, mincut_loss3), (ortho_loss1, ortho_loss2, ortho_loss3)
        
class GAE_Decoder(torch.nn.Module):
    def __init__(self, args):
        super(GAE_Decoder, self).__init__()
        self.conv1 = GCNConv(args.hidden_dim, args.hidden_dim)
        self.conv2 = GCNConv(args.hidden_dim, args.hidden_dim)
        self.conv3 = GCNConv(args.hidden_dim, args.in_dim)
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        
    def forward(self, x3_bar, adj3, Ss):
        x3_dash = x3_bar@Ss[2]
        adj3_dash = Ss[2]@adj3@Ss[2].T
        edge_index3, edge_attr3 = to_edge_index(adj3_dash)
        x2_bar = F.relu(self.conv1(x3_dash, edge_index3, edge_attr3))
        
        x2_dash = x2_bar@Ss[1]
        adj2_dash = Ss[1]@adj3_dash@Ss[1].T
        edge_index2, edge_attr2 = to_edge_index(adj2_dash)
        x1_bar = F.relu(self.conv2(x2_dash, edge_index2, edge_attr2))
        
        x1_dash = x1_bar@Ss[0]
        adj1_dash = Ss[0]@adj2_dash@Ss[0].T
        edge_index1, edge_attr1 = to_edge_index(adj1_dash)
        x_bar = F.relu(self.conv3(x1_dash, edge_index1, edge_attr1))
        
        return x_bar
        
class Classify_with_GAE(torch.nn.Module):
    def __init__(self, args):
        super(Classify_with_GAE, self).__init__()
        self.encoder = GAE_Encoder(args)
        self.decoder = GAE_Decoder(args)
        self.classifier = torch.nn.Linear(args.n3, args.out_dim)
        
    def forward(self, graph):
        x3_bar, adj3, Ss, mincut_losses, ortho_losses = self.encoder(graph.x, graph.edge_index, graph.edge_attr)
        
        class_prob = F.softmax(self.classifier(x3_bar), dim=1)
        
        x_bar = self.decoder(x3_bar, adj3, Ss)
        
        return class_prob, x_bar, mincut_losses, ortho_losses
        
        