import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from sklearn.decomposition import PCA
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

def pca(adata, use_reps=None, n_comps=10):

    """Dimension reduction with PCA algorithm"""
    
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else: 
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray()) 
       else:   
          feat_pca = pca.fit_transform(adata.X)
    
    return feat_pca

def clr_normalize_each_cell(adata, inplace=True):
    
    """Normalize count vector for each cell, i.e. for each row of .X"""

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata     

def buildGraph(adata_omics1, adata_omics2, k = 6):

    """Build graph by spatial coordinate with K-NN"""

    coords = adata_omics1.obsm['spatial'] #Extract spatial coordinate
    features_omics1 = adata_omics1.obsm['feat'] #Extract features of omics 1
    features_omics2 = adata_omics2.obsm['feat'] #Extract features of omics 2

    #Fit KNN model using spatial coordinates
    #Each points will connect to its k nearest neighbors
    knn = NearestNeighbors(n_neighbors = k).fit(coords)

    #Build spares adjacency matrix
    edge_index_np = knn.kneighbors_graph(coords, mode = 'connectivity').tocoo()

    #Convert to pytorch edge_index and make the graph undirected
    edge_index = torch.tensor(np.vstack((
        np.concatenate([edge_index_np.row, edge_index_np.col]),
        np.concatenate([edge_index_np.col, edge_index_np.row]))),
        dtype = torch.long)
        
    #Convert features to tensor
    X_omics1 = torch.tensor(features_omics1, dtype = torch.float)
    X_omics2 = torch.tensor(features_omics2, dtype = torch.float)

    #Create data object
    data = Data()
    data.num_nodes = X_omics1.shape[0]
    data.edge_index = edge_index
    data.x_omics1 = X_omics1   
    data.x_omics2 = X_omics2 

    return data