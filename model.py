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
    
class Model(nn.Module):
    def __init__(
        self, dropout,
        in_omics1, in_omics2,
        branch_dims=(128, 64),
        fusion_dim=128,
        z_dim=32,
        recon_omics1_dim=None,
        recon_omics2_dim=None,
        recon_spatial_dim=None,   
    ):
        super().__init__()
        # Separate 2 omics encoders
        self.omics1_branch = Encoder(in_omics1, branch_dims, dropout=dropout)
        self.omics2_branch = Encoder(in_omics2, branch_dims, dropout=dropout)

        # Fusion
        fused_in = branch_dims[-1] * 2
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, fusion_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(fusion_dim, z_dim)
        self.logvar = nn.Linear(fusion_dim, z_dim)

        # Optional reconstruction heads (feature decoders)
        self.recon_omics1 = None
        self.recon_omics2 = None
        self.recon_spatial = None

        if recon_omics1_dim is not None:
            self.recon_omics1 = nn.Sequential(
                nn.Linear(z_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, recon_omics1_dim),
            )
        if recon_omics2_dim is not None:
            self.recon_omics2 = nn.Sequential(
                nn.Linear(z_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, recon_omics2_dim),
            )
        if recon_spatial_dim is not None:
            self.recon_spatial = nn.Sequential(
                nn.Linear(z_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, recon_spatial_dim),
            )

    def encode(self, x_omics1, x_omics2, edge_index):
        h_omics1 = self.omics1_branch(x_omics1, edge_index)
        h_omics2 = self.omics2_branch(x_omics2, edge_index)
        h = torch.cat([h_omics1, h_omics2], dim=-1)
        h = self.fuse(h)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_graph(self, z, edge_index):
        # Inner-product VGAE decoder
        return torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1))

    def forward(self, x_omics1, x_omics2, edge_index):
        mu, logvar = self.encode(x_omics1, x_omics2, edge_index)
        z = self.reparam(mu, logvar)

        # feature reconstructions
        xhat_omics1 = self.recon_omics1(z) if self.recon_omics1 else None
        xhat_omics2 = self.recon_omics2(z) if self.recon_omics2 else None
        xhat_spatial = self.recon_spatial(z) if self.recon_spatial else None

        # graph reconstruction (edge probabilities)
        adj_pred = self.decode_graph(z, edge_index)

        return z, mu, logvar, xhat_omics1, xhat_omics2, xhat_spatial, adj_pred
