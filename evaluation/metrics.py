"""
evaluation/metrics.py — Clustering quality evaluation.

Internal metrics (no ground truth required):
  Silhouette score  : how well-separated clusters are  [-1, +1]
  Calinski-Harabasz : ratio of between- to within-cluster variance
  Davies-Bouldin    : average cluster similarity (lower = better)
  Modularity Q      : fraction of intra-community edges vs random expectation

External metrics (require ground-truth labels):
  ARI  : Adjusted Rand Index   [-1, +1]   (1 = perfect)
  NMI  : Normalised Mutual Information [0, 1]
  V-measure : harmonic mean of homogeneity and completeness
"""

from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    homogeneity_score,
    completeness_score,
)


def internal_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute clustering quality metrics that don't need ground truth."""
    n_clusters = len(set(labels.tolist()))
    if n_clusters < 2 or n_clusters >= len(X):
        return {"silhouette": float("nan"), "calinski_harabasz": float("nan"),
                "davies_bouldin": float("nan")}
    return {
        "silhouette":        round(float(silhouette_score(X, labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 2),
        "davies_bouldin":    round(float(davies_bouldin_score(X, labels)), 4),
        "n_clusters":        n_clusters,
    }


def external_metrics(
    labels_pred: np.ndarray,
    labels_true: np.ndarray,
) -> Dict[str, float]:
    """Compute clustering quality metrics against ground-truth labels."""
    return {
        "ari":          round(float(adjusted_rand_score(labels_true, labels_pred)), 4),
        "nmi":          round(float(normalized_mutual_info_score(labels_true, labels_pred)), 4),
        "v_measure":    round(float(v_measure_score(labels_true, labels_pred)), 4),
        "homogeneity":  round(float(homogeneity_score(labels_true, labels_pred)), 4),
        "completeness": round(float(completeness_score(labels_true, labels_pred)), 4),
    }


def print_metrics_table(
    results: Dict[str, Dict[str, float]],
    title: str = "Clustering Evaluation",
):
    """Print a formatted comparison table across methods."""
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")
    all_keys = sorted({k for m in results.values() for k in m})
    header = f"  {'Method':<22}" + "".join(f"{k:>12}" for k in all_keys)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for method, metrics in results.items():
        row = f"  {method:<22}"
        for k in all_keys:
            v = metrics.get(k, float("nan"))
            row += f"{v:>12.4f}" if not isinstance(v, int) else f"{v:>12d}"
        print(row)
    print(f"{'═'*60}\n")
