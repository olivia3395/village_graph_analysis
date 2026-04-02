"""
graph/features.py — Graph-structural node feature extraction.

Computes structural properties of each node in the village graph
that encode its position and role in the network. These features
complement the socio-economic attributes for clustering.

Structural features
───────────────────
  degree           : number of connected neighbours
  weighted_degree  : sum of edge weights (connectivity strength)
  clustering_coef  : fraction of neighbours that are also connected
  betweenness      : fraction of shortest paths passing through this node
                     (high = bridge/gateway village)
  closeness        : inverse of average distance to all other villages
  pagerank         : importance relative to neighbours' importance
  local_density    : mean weight of edges within 1-hop neighbourhood
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
import networkx as nx
import pandas as pd


def extract_graph_features(G: nx.Graph) -> pd.DataFrame:
    """
    Compute structural graph features for every node.

    Args:
        G: weighted undirected village graph

    Returns:
        DataFrame indexed by village_id with structural feature columns
    """
    nodes = list(G.nodes())

    degree       = dict(G.degree())
    w_degree     = dict(G.degree(weight="weight"))
    clustering   = nx.clustering(G, weight="weight")
    pagerank     = nx.pagerank(G, weight="weight", alpha=0.85)

    # Betweenness and closeness can be slow for large graphs
    if len(nodes) <= 500:
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
        closeness   = nx.closeness_centrality(G, distance="weight")
    else:
        betweenness = {n: 0.0 for n in nodes}
        closeness   = {n: 0.0 for n in nodes}

    # Local density: mean edge weight in 1-hop neighbourhood
    local_density = {}
    for n in nodes:
        nbr_weights = [G[n][nb].get("weight", 0.0) for nb in G.neighbors(n)]
        local_density[n] = np.mean(nbr_weights) if nbr_weights else 0.0

    data = []
    for n in nodes:
        data.append({
            "village_id":       n,
            "g_degree":         degree[n],
            "g_weighted_degree": w_degree[n],
            "g_clustering":     clustering[n],
            "g_betweenness":    betweenness[n],
            "g_closeness":      closeness[n],
            "g_pagerank":       pagerank[n],
            "g_local_density":  local_density[n],
        })

    return pd.DataFrame(data).set_index("village_id")


def get_adjacency_matrix(G: nx.Graph, village_ids: List) -> np.ndarray:
    """Return the weighted adjacency matrix in node order given by village_ids."""
    return nx.to_numpy_array(G, nodelist=village_ids, weight="weight")


def get_laplacian(A: np.ndarray, normalised: bool = True) -> np.ndarray:
    """
    Compute the (normalised) graph Laplacian L = I - D^{-1/2} A D^{-1/2}.

    The Laplacian's eigenvectors encode the global structure of the graph
    and are used as features for spectral clustering.
    """
    d  = A.sum(axis=1)
    if normalised:
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = np.eye(len(d)) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        L = np.diag(d) - A
    return L
