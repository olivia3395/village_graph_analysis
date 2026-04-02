"""
graph/builder.py — Build the village graph from multi-source indicators.

Graph representation
─────────────────────
  Nodes  : villages (one per row in the feature matrix)
  Edges  : connect pairs of villages that are spatially proximate
            AND/OR feature-similar
  Weights: combination of spatial and feature-space similarity

Edge construction strategies
─────────────────────────────
  "spatial"  → edge if geodesic distance < threshold_km
  "knn"      → k-nearest neighbours by cosine similarity in feature space
  "hybrid"   → weighted combination:
                  w(i,j) = α · spatial_sim(i,j) + (1-α) · feature_sim(i,j)

Why hybrid?
───────────
Purely spatial graphs miss socio-economic linkages between distant but
similar villages (e.g. two underdeveloped villages in different corners
of the county). Purely feature-based graphs ignore geographic realities
(planning boundaries, road connectivity). The hybrid captures both.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------------

def haversine_km(
    lon1: float, lat1: float,
    lon2: float, lat2: float,
) -> float:
    """Great-circle distance in km between two (lon, lat) points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def spatial_distance_matrix(
    lons: np.ndarray, lats: np.ndarray
) -> np.ndarray:
    """Compute N×N matrix of pairwise haversine distances (km)."""
    N = len(lons)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = haversine_km(lons[i], lats[i], lons[j], lats[j])
            D[i, j] = D[j, i] = d
    return D


def feature_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute N×N cosine similarity matrix from feature matrix X.
    Values in [0, 1] (cosine similarity shifted to non-negative range).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    X_norm = X / norms
    cos_sim = X_norm @ X_norm.T
    # Shift from [-1,1] to [0,1]
    return (cos_sim + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class VillageGraphBuilder:
    """
    Constructs a weighted NetworkX graph over villages.

    Usage:
        builder = VillageGraphBuilder(cfg.graph)
        G = builder.build(df, X)
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def build(
        self,
        df,           # DataFrame with 'longitude', 'latitude' columns
        X: np.ndarray,  # (N, F) normalised feature matrix
        village_ids=None,
    ) -> nx.Graph:
        """
        Build the village graph.

        Args:
            df         : village DataFrame (with lon/lat columns)
            X          : normalised feature matrix (N, F)
            village_ids: list of node IDs (defaults to df.index)

        Returns:
            G: weighted undirected NetworkX graph
               Nodes have attributes: lon, lat, features
               Edges have attributes: weight, spatial_sim, feature_sim, distance_km
        """
        N = len(df)
        ids = village_ids if village_ids is not None else list(df.index)

        lons = df["longitude"].values
        lats = df["latitude"].values

        print(f"\n[Graph Builder]  strategy={self.cfg.edge_strategy}  N={N}")

        # ── Compute similarity matrices ────────────────────────────────────
        print("  Computing spatial distances...")
        D_spatial = spatial_distance_matrix(lons, lats)          # (N, N) km
        # Convert distance → similarity: exp(-d / sigma)
        sigma = self.cfg.distance_threshold_km / 2.0
        S_spatial = np.exp(-D_spatial / sigma)
        np.fill_diagonal(S_spatial, 0.0)

        print("  Computing feature similarities...")
        S_feature = feature_similarity_matrix(X)                 # (N, N) in [0,1]
        np.fill_diagonal(S_feature, 0.0)

        # ── Combined weight matrix ─────────────────────────────────────────
        alpha = self.cfg.spatial_weight
        W = alpha * S_spatial + (1 - alpha) * S_feature

        # ── Build NetworkX graph ───────────────────────────────────────────
        G = nx.Graph()

        # Add nodes with attributes
        for i, vid in enumerate(ids):
            node_features = {col: float(df.iloc[i][col])
                             for col in df.columns if col not in ("longitude","latitude")}
            G.add_node(vid, lon=lons[i], lat=lats[i],
                       index=i, **node_features)

        # Add edges according to strategy
        n_edges = 0
        if self.cfg.edge_strategy in ("spatial", "hybrid"):
            for i in range(N):
                for j in range(i+1, N):
                    dist = D_spatial[i, j]
                    if dist <= self.cfg.distance_threshold_km:
                        w = float(W[i, j])
                        if w >= self.cfg.min_edge_weight:
                            G.add_edge(ids[i], ids[j],
                                       weight=w,
                                       spatial_sim=float(S_spatial[i, j]),
                                       feature_sim=float(S_feature[i, j]),
                                       distance_km=float(dist))
                            n_edges += 1

        elif self.cfg.edge_strategy == "knn":
            k = self.cfg.knn_k
            for i in range(N):
                knn = np.argsort(S_feature[i])[::-1][1:k+1]
                for j in knn:
                    if not G.has_edge(ids[i], ids[j]):
                        w = float(W[i, j])
                        if w >= self.cfg.min_edge_weight:
                            G.add_edge(ids[i], ids[j],
                                       weight=w,
                                       spatial_sim=float(S_spatial[i, j]),
                                       feature_sim=float(S_feature[i, j]),
                                       distance_km=float(D_spatial[i, j]))
                            n_edges += 1

        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"  Avg degree: {2*G.number_of_edges()/max(G.number_of_nodes(),1):.1f}")
        print(f"  Connected components: {nx.number_connected_components(G)}")

        # Store weight matrix as graph attribute for spectral methods
        G.graph["W"] = W
        G.graph["D_spatial"] = D_spatial
        G.graph["village_ids"] = ids

        return G
