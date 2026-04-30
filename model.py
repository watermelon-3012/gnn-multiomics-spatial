from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn

class Encoder(nn.Module): 
    def __init__(self, in_dim, hidden_dims, dropout): 
        super().__init__() 
        self.gcn_layers = nn.ModuleList()
        last = in_dim
        for h in hidden_dims: 
            self.gcn_layers.append(GCNConv(last, h)) 
            last = h 
        self.dropout = dropout
            
    def forward(self, x, edge_index):
        h = x
        for conv in self.gcn_layers:
            h = conv(h, edge_index)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    

