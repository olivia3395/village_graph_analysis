"""
models/community.py — Graph community detection methods.

Methods
───────
  Louvain       : Greedy modularity optimisation. Fast, scalable.
                  Directly optimises Q = Σ_c [L_c/m - (d_c/2m)²]
                  where L_c = intra-community edges, d_c = community degree sum.

  Girvan-Newman : Iteratively removes highest-betweenness edges.
                  Slower but produces a full dendrogram.

For village planning, Louvain is preferred because:
  - Runtime: O(n log n) vs O(n³) for G-N
  - Resolution parameter allows tuning cluster granularity
  - Communities map naturally to planning zones
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
import networkx as nx


class LouvainDetector:
    """
    Louvain community detection via modularity maximisation.

    Falls back to a greedy modularity algorithm if the `community`
    package is not installed.
    """

    def __init__(self, cfg):
        self.cfg         = cfg
        self.communities_: List[set] = []
        self.labels_:      np.ndarray = None
        self.modularity_:  float = 0.0

    def fit_predict(self, G, village_ids) -> np.ndarray:
        """
        Detect communities and return per-node label array.

        Returns:
            labels: (N,) integer community assignments
        """
        print(f"\n[Louvain Community Detection]  resolution={self.cfg.louvain_resolution}")

        # Try python-louvain or networkx greedy_modularity
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(
                G,
                weight="weight",
                resolution=self.cfg.louvain_resolution,
                random_state=self.cfg.louvain_seed,
            )
            self.modularity_ = community_louvain.modularity(partition, G, weight="weight")
            raw_labels = [partition.get(vid, 0) for vid in village_ids]

        except ImportError:
            # Fallback: networkx greedy modularity
            communities_gen = nx.community.greedy_modularity_communities(
                G, weight="weight"
            )
            communities = list(communities_gen)
            partition = {}
            for cid, comm in enumerate(communities):
                for node in comm:
                    partition[node] = cid
            self.modularity_ = nx.community.modularity(G, communities, weight="weight")
            raw_labels = [partition.get(vid, 0) for vid in village_ids]

        labels = np.array(raw_labels, dtype=int)

        # Remap to contiguous integers
        unique_labels = sorted(set(labels.tolist()))
        remap = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([remap[l] for l in labels.tolist()])

        self.labels_ = labels
        n_comm = len(set(labels.tolist()))
        print(f"  Found {n_comm} communities  (modularity Q={self.modularity_:.4f})")
        _print_label_dist(labels)
        return labels


class GirvanNewmanDetector:
    """
    Girvan-Newman hierarchical community detection.

    Produces a dendrogram; we cut at the target number of communities.
    Best for small graphs (N < 200) due to O(N³) complexity.
    """

    def __init__(self, cfg):
        self.cfg    = cfg
        self.labels_: np.ndarray = None

    def fit_predict(self, G, village_ids) -> np.ndarray:
        k = self.cfg.n_clusters
        print(f"\n[Girvan-Newman Detection]  k={k}")

        if len(village_ids) > 300:
            print("  Warning: GN is slow for N>300. Consider Louvain instead.")

        comp_gen = nx.community.girvan_newman(G)
        communities = None
        for communities in comp_gen:
            if len(communities) >= k:
                break

        if communities is None:
            # Fallback: all in one community
            labels = np.zeros(len(village_ids), dtype=int)
        else:
            partition = {}
            for cid, comm in enumerate(communities):
                for node in comm:
                    partition[node] = cid
            labels = np.array([partition.get(vid, 0) for vid in village_ids])

        self.labels_ = labels
        print(f"  Found {len(set(labels.tolist()))} communities")
        _print_label_dist(labels)
        return labels


def _print_label_dist(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Community sizes: {dict(zip(unique.tolist(), counts.tolist()))}")
