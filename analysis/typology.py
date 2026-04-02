"""
analysis/typology.py — Village typology identification.

After clustering, we interpret each cluster as a village typology
by examining the mean feature values per cluster and matching them
to the domain-defined archetypes:

  Agricultural Core     : high cultivated_area, mechanisation, irrigation
  Tourism & Cultural    : high tourism_revenue, elevation, scenic diversity
  Peri-urban Transition : low dist_to_county, high road_quality, broadband
  Underdeveloped Priority: low income, high elderly_ratio, low infrastructure
  Industrial & Commercial: high income, industry_diversity, market_access
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Archetype signatures: feature → expected direction for this type
# +1 = high value expected, -1 = low value expected
ARCHETYPE_SIGNATURES: Dict[str, Dict[str, int]] = {
    "Agricultural Core": {
        "cultivated_area_mu": +1, "mechanisation_rate": +1,
        "crop_diversity": +1,     "irrigation_coverage": +1,
        "dist_to_county_km": 0,   "tourism_revenue_cny": -1,
    },
    "Tourism & Cultural": {
        "tourism_revenue_cny": +1, "elevation_m": +1,
        "industry_diversity": +1,  "income_per_capita_cny": +1,
        "cultivated_area_mu": -1,
    },
    "Peri-urban Transition": {
        "dist_to_county_km": -1, "road_quality": +1,
        "broadband_coverage": +1, "population": +1,
        "income_per_capita_cny": +1,
    },
    "Underdeveloped Priority": {
        "income_per_capita_cny": -1, "elderly_ratio": +1,
        "road_quality": -1,          "broadband_coverage": -1,
        "education_index": -1,       "healthcare_access": -1,
    },
    "Industrial & Commercial": {
        "income_per_capita_cny": +1, "industry_diversity": +1,
        "market_access_score": +1,   "employment_rate": +1,
        "population": +1,
    },
}


def label_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    village_ids: List[str],
    typology_cfg,
) -> Dict[int, str]:
    """
    Assign a typology name to each cluster by matching cluster-mean
    feature profiles against archetype signatures.

    Args:
        df          : normalised village DataFrame
        labels      : (N,) cluster assignment array
        village_ids : list of village IDs matching label order
        typology_cfg: TypologyConfig

    Returns:
        cluster_to_typology: dict of cluster_id → typology name string
    """
    df = df.copy()
    df["cluster"] = labels

    n_clusters   = len(set(labels.tolist()))
    cluster_means = df.groupby("cluster").mean()

    cluster_to_typology: Dict[int, str] = {}
    used_typologies = set()

    scores_matrix = {}
    for cid in range(n_clusters):
        if cid not in cluster_means.index:
            continue
        row = cluster_means.loc[cid]
        scores_matrix[cid] = {}
        for tname, sig in ARCHETYPE_SIGNATURES.items():
            score = 0.0
            for feat, direction in sig.items():
                if feat in row.index:
                    score += direction * float(row[feat])
            scores_matrix[cid][tname] = score

    # Greedy matching: assign each cluster to the best-matching unused typology
    assigned = {}
    remaining_clusters = list(range(n_clusters))

    for _ in range(n_clusters):
        if not remaining_clusters:
            break
        best_pair  = None
        best_score = -1e9
        for cid in remaining_clusters:
            if cid not in scores_matrix:
                continue
            for tname, sc in scores_matrix[cid].items():
                if tname not in used_typologies and sc > best_score:
                    best_score = sc
                    best_pair  = (cid, tname)
        if best_pair is None:
            break
        cid, tname = best_pair
        assigned[cid] = tname
        used_typologies.add(tname)
        remaining_clusters.remove(cid)

    # Fill any unassigned clusters
    all_names = list(ARCHETYPE_SIGNATURES.keys())
    fallback  = [n for n in all_names if n not in used_typologies]
    for cid in remaining_clusters:
        name = fallback.pop(0) if fallback else f"Type_{cid}"
        assigned[cid] = name

    return assigned


def typology_profile(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_to_typology: Dict[int, str],
    key_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Return a summary table of mean feature values per typology.

    Args:
        df                  : normalised village DataFrame
        labels              : (N,) cluster assignments
        cluster_to_typology : output from label_clusters()
        key_features        : subset of features to include (None = all)

    Returns:
        profile: DataFrame with typologies as index and features as columns
    """
    df = df.copy()
    df["cluster"]  = labels
    df["typology"] = [cluster_to_typology.get(l, f"Type_{l}") for l in labels]

    feats = key_features or [
        "income_per_capita_cny", "employment_rate", "tourism_revenue_cny",
        "road_quality", "broadband_coverage", "education_index",
        "healthcare_access", "cultivated_area_mu", "mechanisation_rate",
        "dist_to_county_km", "population", "elderly_ratio",
        "development_index", "revitalization_need",
    ]
    feats = [f for f in feats if f in df.columns]
    profile = df.groupby("typology")[feats].mean().round(3)
    return profile
