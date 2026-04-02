"""
visualization/plots.py — Visualisation for village graph analysis.

Plots
─────
  plot_spatial_clusters   : villages on a map coloured by cluster
  plot_graph              : village graph with edges drawn
  plot_typology_radar     : radar chart of mean features per typology
  plot_development_index  : spatial heatmap of development_index
  plot_dendrogram         : cluster hierarchy (from spectral eigenvalues)
  plot_metrics_comparison : bar chart comparing ARI/NMI across methods
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx


PALETTE = [
    "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",
    "#00BCD4", "#FF5722", "#795548", "#607D8B", "#FFC107",
]


def plot_spatial_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_to_typology: Dict[int, str],
    title: str = "Village Typology Clusters — Ji County",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Scatter plot of villages coloured by cluster on the spatial map."""
    fig, ax = plt.subplots(figsize=(9, 7))
    n_clusters = len(set(labels.tolist()))

    for cid in range(n_clusters):
        mask = labels == cid
        color = PALETTE[cid % len(PALETTE)]
        tname = cluster_to_typology.get(cid, f"Cluster {cid}")
        ax.scatter(
            df.loc[mask, "longitude"] if "longitude" in df.columns else np.zeros(mask.sum()),
            df.loc[mask, "latitude"]  if "latitude"  in df.columns else np.zeros(mask.sum()),
            c=color, label=f"{tname} (n={mask.sum()})",
            s=60, alpha=0.85, edgecolors="white", linewidths=0.5,
        )

    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude",  fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_graph(
    G: nx.Graph,
    labels: np.ndarray,
    village_ids: List[str],
    cluster_to_typology: Dict[int, str],
    title: str = "Village Graph",
    save_path: Optional[str] = None,
    show: bool = True,
    max_nodes: int = 150,
):
    """Draw the village graph with nodes coloured by cluster."""
    if len(village_ids) > max_nodes:
        # Sample for readability
        idx = np.random.choice(len(village_ids), max_nodes, replace=False)
        sub_ids = [village_ids[i] for i in idx]
        G_draw  = G.subgraph(sub_ids)
        sub_lab = labels[idx]
    else:
        G_draw  = G
        sub_ids = village_ids
        sub_lab = labels

    fig, ax = plt.subplots(figsize=(10, 8))
    node_colors = [PALETTE[sub_lab[i] % len(PALETTE)] for i in range(len(sub_ids))]

    # Use spatial positions if available
    pos = {}
    for i, vid in enumerate(sub_ids):
        node = G_draw.nodes[vid]
        pos[vid] = (node.get("lon", 0), node.get("lat", 0))

    edge_weights = [G_draw[u][v].get("weight", 0.1) for u, v in G_draw.edges()]
    nx.draw_networkx_nodes(G_draw, pos, nodelist=sub_ids,
                           node_color=node_colors, node_size=40,
                           alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G_draw, pos,
                           width=[w * 1.5 for w in edge_weights],
                           alpha=0.2, edge_color="gray", ax=ax)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_typology_radar(
    profile: pd.DataFrame,
    title: str = "Village Typology Feature Profiles",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Radar (spider) chart showing mean feature values per typology."""
    feats = [c for c in profile.columns if not c.startswith("g_")][:8]
    if not feats:
        return

    angles = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    typologies = profile.index.tolist()

    for i, typ in enumerate(typologies):
        values = profile.loc[typ, feats].tolist()
        values += values[:1]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=typ)
        ax.fill(angles, values, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace("_", "\n")[:12] for f in feats], size=8)
    ax.set_title(title, size=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_development_heatmap(
    df: pd.DataFrame,
    column: str = "development_index",
    title: str = "Village Development Index",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Spatial heatmap coloured by a development score."""
    if "longitude" not in df.columns or "latitude" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        df["longitude"], df["latitude"],
        c=df[column] if column in df.columns else 0.5,
        cmap="RdYlGn", s=80, alpha=0.9,
        edgecolors="white", linewidths=0.4,
        vmin=0, vmax=1,
    )
    plt.colorbar(sc, ax=ax, label=column.replace("_", " ").title())
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    metrics_to_plot: List[str] = ("ari", "nmi", "silhouette"),
    title: str = "Clustering Method Comparison",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Bar chart comparing clustering methods across quality metrics."""
    methods = list(results.keys())
    n_metrics = len(metrics_to_plot)
    x = np.arange(len(methods))
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, metric in enumerate(metrics_to_plot):
        vals = [results[m].get(metric, 0.0) for m in methods]
        ax.bar(x + i * width, vals, width,
               label=metric.upper(), color=PALETTE[i], alpha=0.85, edgecolor="white")

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def _save_or_show(fig, save_path, show):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)
