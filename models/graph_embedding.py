"""
models/graph_embedding.py — Graph embedding via random walks (Node2Vec style).

Node2Vec
─────────
Generates biased random walks on the village graph, then trains a
skip-gram model (Word2Vec) to embed nodes into R^d such that
structurally similar nodes have similar embeddings.

Two hyperparameters control the walk behaviour:
  p : return parameter (low p → DFS, explores structure)
  q : in-out parameter (low q → BFS, explores local neighbourhood)

For village planning we use q < 1 (DFS-biased) to capture
global structural roles: a "gateway" village bridges two clusters
regardless of which specific cluster it is in.

After embedding, we apply k-means to the d-dimensional space.
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np
import networkx as nx
from sklearn.cluster import KMeans


class Node2VecEmbedder:
    """
    Lightweight Node2Vec implementation using NetworkX random walks
    and a simple co-occurrence embedding (no gensim dependency).

    For production use, replace with gensim.models.Word2Vec or
    the `node2vec` package for better performance.
    """

    def __init__(self, cfg):
        self.cfg       = cfg
        self.embedding: Optional[np.ndarray] = None
        self.labels_:   Optional[np.ndarray] = None

    def _biased_walk(
        self, G: nx.Graph, start: str, length: int, p: float, q: float
    ) -> List[str]:
        """Generate one biased random walk from a start node."""
        walk = [start]
        while len(walk) < length:
            curr = walk[-1]
            nbrs = list(G.neighbors(curr))
            if not nbrs:
                break
            if len(walk) == 1:
                nxt = np.random.choice(nbrs)
            else:
                prev  = walk[-2]
                probs = []
                for nb in nbrs:
                    if nb == prev:
                        probs.append(1.0 / p)
                    elif G.has_edge(prev, nb):
                        probs.append(1.0)
                    else:
                        probs.append(1.0 / q)
                probs = np.array(probs)
                probs /= probs.sum()
                nxt = np.random.choice(nbrs, p=probs)
            walk.append(nxt)
        return walk

    def _build_cooccurrence(
        self, walks: List[List[str]], node2idx: dict, window: int
    ) -> np.ndarray:
        """Build a co-occurrence matrix from random walks."""
        N = len(node2idx)
        C = np.zeros((N, N), dtype=np.float32)
        for walk in walks:
            for i, node in enumerate(walk):
                nid = node2idx[node]
                for j in range(max(0, i-window), min(len(walk), i+window+1)):
                    if i != j:
                        nbr_id = node2idx[walk[j]]
                        C[nid, nbr_id] += 1.0
        return C

    def fit(self, G: nx.Graph, village_ids: List[str]):
        """
        Learn node embeddings via random walks + SVD on co-occurrence.
        """
        cfg = self.cfg
        d   = cfg.n2v_dimensions
        print(f"\n[Node2Vec Embedding]  d={d}  walks={cfg.n2v_num_walks}  len={cfg.n2v_walk_length}")

        np.random.seed(42)
        node2idx = {v: i for i, v in enumerate(village_ids)}

        # Generate random walks
        walks = []
        for _ in range(cfg.n2v_num_walks):
            np.random.shuffle(village_ids)
            for node in village_ids:
                walk = self._biased_walk(
                    G, node, cfg.n2v_walk_length,
                    p=cfg.n2v_p, q=cfg.n2v_q,
                )
                walks.append(walk)

        # Build co-occurrence matrix and apply SVD
        C = self._build_cooccurrence(walks, node2idx, cfg.n2v_window)
        # PPMI (Positive Pointwise Mutual Information) weighting
        total     = C.sum()
        row_sums  = C.sum(axis=1, keepdims=True)
        col_sums  = C.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ppmi = np.log(np.where(C > 0, C * total / (row_sums * col_sums + 1e-12), 1e-12))
        ppmi = np.maximum(ppmi, 0.0)

        # SVD → embedding
        U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
        dim  = min(d, U.shape[1])
        emb  = U[:, :dim] * np.sqrt(S[:dim])
        # L2-normalise
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb   = emb / np.where(norms > 0, norms, 1.0)

        self.embedding = emb
        print(f"  Embedding shape: {emb.shape}")
        return self

    def predict(self, k: int) -> np.ndarray:
        """K-means on the embedding space."""
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        self.labels_ = km.fit_predict(self.embedding)
        unique, counts = np.unique(self.labels_, return_counts=True)
        print(f"  N2V cluster sizes: {dict(zip(unique.tolist(), counts.tolist()))}")
        return self.labels_

    def fit_predict(self, G: nx.Graph, village_ids: List[str]) -> np.ndarray:
        self.fit(G, village_ids)
        return self.predict(self.cfg.n_clusters)
