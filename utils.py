import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.utils import negative_sampling
from scipy.spatial import cKDTree
from scipy.stats import mode
from collections import Counter
from torch_geometric.utils import negative_sampling, to_undirected
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score

def edge_recon_loss(z, edge_index, num_nodes, neg_ratio=1.0):

    """Loss function for Edge"""

    pos_edge_index = edge_index
    num_pos = pos_edge_index.size(1)
    num_neg = int(neg_ratio * num_pos)

    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg,
        method="sparse"
    )

    pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
    neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)

    pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
    neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
    return pos_loss + neg_loss

def kl_loss(mu, logvar, eps=1e-9):

    """KL Loss"""

    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp() + eps)

def train_model(model, data, epochs, device,
               lr=1e-3, weight_decay=1e-5, dropout=0.2,
               lambda_omics1=1.0, lambda_omics2=1.0):


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    edge_index = data.edge_index.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        z, mu, logvar, xhat_omics1, xhat_omics2, _, adj_pred = model(
            data.x_omics1.to(device),
            data.x_omics2.to(device),
            edge_index
        )

        # Graph reconstruction loss (pos + neg edges)
        pos_pred = model.decode_graph(z, edge_index)
        pos_label = torch.ones_like(pos_pred)

        # sample negative edges
        neg_edge_index = negative_sampling(
            edge_index, num_nodes=z.size(0), num_neg_samples=edge_index.size(1)
        )
        neg_pred = model.decode_graph(z, neg_edge_index)
        neg_label = torch.zeros_like(neg_pred)

        # combine pos & neg
        preds = torch.cat([pos_pred, neg_pred])
        labels = torch.cat([pos_label, neg_label])
        loss_edges = F.binary_cross_entropy(preds, labels)

        # KL loss 
        loss_kld = kl_loss(mu, logvar)

        # Feature reconstruction losses 
        loss_omics1 = F.mse_loss(xhat_omics1, data.x_omics1.to(device)) if xhat_omics1 is not None else 0
        loss_omics2 = F.mse_loss(xhat_omics2, data.x_omics2.to(device)) if xhat_omics2 is not None else 0

        # Total loss
        loss = loss_edges + loss_kld + lambda_omics1 * loss_omics1 + lambda_omics2 * loss_omics2

        # Backprop
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1:03d} | "
              f"Total: {loss.item():.4f} | "
              f"Graph: {loss_edges.item():.4f} | "
              f"KLD: {loss_kld.item():.4f} | "
              f"Omics1: {float(loss_omics1):.4f} | "
              f"Omics2: {float(loss_omics2):.4f}")

    # Final embeddings
    model.eval()
    with torch.no_grad():
        z, _, _, _, _, _, _ = model(
            data.x_omics1.to(device),
            data.x_omics2.to(device),
            edge_index
        )

    return z

def cKD_refine_label(coords, labels, k = 50):

    """
    Refine labels after clustering
    
    coord: Coordinates of cells
    labels: Cluster result
    k: number of closest point of each cell
    """

    # Step 1: Build KD-Tree
    tree = cKDTree(coords.copy())
    # Step 2: Find k-nearest neighbors for each spot
    # k+1 because the closest point is itself
    distances, neighbors = tree.query(coords, k=k+1)
    # Exclude self-neighbor (first column)
    neighbors = neighbors[:, 1:]
    # Step 3: Reassign labels
    new_labels = labels.copy()
    for i, nbrs in enumerate(neighbors):
        # Get the labels of neighboring spots
        neighbor_labels = labels[nbrs]
        # Find the most common label among neighbors
        # most_common_label = mode(neighbor_labels).mode[0]
        most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
        # Reassign the label
        new_labels[i] = most_common_label
    return (new_labels)

def clustering(z, label, coords, k,
               covariance_type='full', 
               tol=0.001, 
               reg_covar=1e-06, 
               max_iter=100, 
               n_init=1, 
               init_params='kmeans', 
               weight_concentration_prior_type='dirichlet_process', 
               weight_concentration_prior=None, 
               mean_precision_prior=None, 
               mean_prior=None, 
               degrees_of_freedom_prior=None, 
               covariance_prior=None, 
               random_state=None, 
               warm_start=False, 
               verbose=0, 
               verbose_interval=10):
    
    """
    z: Model output
    label: Column of label in the AnnData object
    coords: Column of spatial coordinates in the AnnData object
    k: k value for cKD refine label
    """

    # detach model output to CPU numpy
    z_np = z.detach().cpu().numpy()
    df_z = pd.DataFrame(z_np)

    num_cluster = len(label.unique())
    bayes = BayesianGaussianMixture(num_cluster,
                                    covariance_type, 
                                    tol, 
                                    reg_covar, 
                                    max_iter, 
                                    n_init, 
                                    init_params, 
                                    weight_concentration_prior_type, 
                                    weight_concentration_prior, 
                                    mean_precision_prior, 
                                    mean_prior, 
                                    degrees_of_freedom_prior, 
                                    covariance_prior, 
                                    random_state, 
                                    warm_start, 
                                    verbose, 
                                    verbose_interval, 
                                )
    bayes.fit(df_z)
    cluster_bayes = bayes.predict(df_z)
    pred = cKD_refine_label(np.array(coords), cluster_bayes, k)

    return pred

def compute_metrics(labels_true, labels_pred):

    """
    labels_true: ground truth
    labels_pred: prediction
    """

    ARI = adjusted_rand_score(labels_true, labels_pred)
    NMI = normalized_mutual_info_score(labels_true, labels_pred)
    HOM = homogeneity_score(labels_true, labels_pred)

    return {"ARI": ARI, "NMI": NMI, "HOM": HOM}