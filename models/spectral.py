"""
models/spectral.py — Spectral clustering for village typology detection.

Theory
──────
Spectral clustering embeds villages into a low-dimensional Euclidean space
using the eigenvectors of the graph Laplacian, then applies k-means there.

Steps:
  1. Compute normalised Laplacian  L = I - D^{-1/2} A D^{-1/2}
  2. Compute the k smallest eigenvectors of L  → embedding matrix U ∈ R^{N×k}
  3. Row-normalise U (each village → unit vector on the k-sphere)
  4. Run k-means on U → cluster assignments

Why spectral clustering for villages?
──────────────────────────────────────
  • Handles non-convex clusters (mountain villages form irregular shapes)
  • Uses the graph structure directly (spatial + socio-economic edges)
  • Consistent even when clusters are imbalanced in size
  • The Fiedler vector (2nd eigenvector) alone gives a meaningful 1-D
    development gradient that planners can read directly
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from graph.features import get_adjacency_matrix, get_laplacian


class SpectralClusterer:
    """
    Spectral clustering using the village graph Laplacian.

    Usage:
        sc     = SpectralClusterer(cfg.clustering)
        labels = sc.fit_predict(G, village_ids)
        U      = sc.embedding   # (N, k) spectral embedding
    """

    def __init__(self, cfg):
        self.cfg       = cfg
        self.embedding: Optional[np.ndarray] = None
        self.labels_:   Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None

    def fit_predict(self, G, village_ids) -> np.ndarray:
        """
        Run spectral clustering on the village graph.

        Returns:
            labels: (N,) integer cluster assignments
        """
        k = self.cfg.n_clusters
        print(f"\n[Spectral Clustering]  k={k}")

        # ── Build Laplacian ────────────────────────────────────────────────
        A = get_adjacency_matrix(G, village_ids)
        L = get_laplacian(A, normalised=True)

        # ── Eigen-decomposition ────────────────────────────────────────────
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Take the k smallest (they correspond to the smoothest graph signals)
        idx = np.argsort(eigenvalues)
        self.eigenvalues = eigenvalues[idx]
        U = eigenvectors[:, idx[1:k+1]]   # skip the trivial zero eigenvector

        # ── Row-normalise (standard for spectral clustering) ───────────────
        U_norm = normalize(U, norm="l2")
        self.embedding = U_norm

        # ── k-means on the embedding ───────────────────────────────────────
        km = KMeans(
            n_clusters=k,
            n_init=self.cfg.spectral_n_init,
            random_state=self.cfg.spectral_random_state,
        )
        labels = km.fit_predict(U_norm)
        self.labels_ = labels

        print(f"  Eigenvalue gap (λ_{k} - λ_{k+1}): "
              f"{self.eigenvalues[k]:.4f} - {self.eigenvalues[k+1]:.4f} "
              f"= {self.eigenvalues[k+1]-self.eigenvalues[k]:.4f}")
        _print_label_dist(labels, k)
        return labels

    def fiedler_vector(self) -> np.ndarray:
        """The Fiedler vector (2nd eigenvector) — a 1-D development gradient."""
        if self.embedding is None:
            raise RuntimeError("Call fit_predict first")
        return self.embedding[:, 0]


def _print_label_dist(labels, k):
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"  Cluster sizes: {dist}")
