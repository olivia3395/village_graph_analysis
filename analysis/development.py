"""
analysis/development.py — Development potential and revitalization priority scoring.

For each village, computes:
  development_score    : current overall development level [0, 1]
  revitalization_need  : urgency of policy support [0, 1]
  growth_potential     : estimated capacity for improvement [0, 1]
  priority_tier        : Tier 1 (urgent) / Tier 2 / Tier 3 (self-sufficient)

These scores directly inform:
  - Allocation of revitalization funds across villages
  - Infrastructure investment sequencing
  - Policy zone designation (国家级 / 省级 / 市级 支持)
"""

from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def compute_development_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute development and revitalization scores for all villages.

    Uses the composite indicators computed during preprocessing.
    If pre-computed columns are missing, falls back to direct computation.
    """
    df = df.copy()

    def _safe(cols, weights=None):
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return pd.Series(0.5, index=df.index)
        if weights:
            w = np.array([weights[c] for c in cols])
            w /= w.sum()
            return sum(df[c] * w[i] for i, c in enumerate(cols))
        return df[cols].mean(axis=1)

    # Economic vitality
    if "economic_vitality" not in df.columns:
        df["economic_vitality"] = _safe([
            "income_per_capita_cny", "employment_rate",
            "market_access_score", "industry_diversity",
        ])

    # Social welfare
    if "social_welfare" not in df.columns:
        df["social_welfare"] = _safe([
            "education_index", "healthcare_access", "youth_ratio",
        ])

    # Infrastructure
    if "infrastructure_score" not in df.columns:
        df["infrastructure_score"] = _safe([
            "road_quality", "broadband_coverage",
            "water_supply", "electricity_reliability",
        ])

    # Overall development index
    if "development_index" not in df.columns:
        df["development_index"] = (
            0.35 * df["economic_vitality"] +
            0.25 * df["social_welfare"] +
            0.25 * df["infrastructure_score"]
        ).clip(0, 1)

    # Revitalization need = inverse of development
    df["revitalization_need"] = (1.0 - df["development_index"]).clip(0, 1)

    # Growth potential: villages with moderate dev + good resources have highest upside
    resource_score = _safe([
        "cultivated_area_mu", "tourism_revenue_cny", "crop_diversity",
    ])
    # Growth potential peaks in the middle of development_index (0.3–0.6)
    d = df["development_index"]
    df["growth_potential"] = (
        resource_score * np.exp(-2.0 * (d - 0.4).abs())
    ).clip(0, 1)

    # Priority tier
    def _tier(row):
        if row["revitalization_need"] > 0.65:
            return "Tier 1 — Urgent"
        elif row["revitalization_need"] > 0.40:
            return "Tier 2 — Moderate"
        else:
            return "Tier 3 — Self-sufficient"

    df["priority_tier"] = df.apply(_tier, axis=1)

    return df


def rank_villages(
    df: pd.DataFrame,
    by: str = "revitalization_need",
    top_n: int = 20,
    ascending: bool = False,
) -> pd.DataFrame:
    """Return the top-N villages ranked by the given score."""
    cols = [by, "priority_tier", "development_index",
            "economic_vitality", "infrastructure_score"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values(by, ascending=ascending).head(top_n)


def cluster_development_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_to_typology: Dict[int, str],
) -> pd.DataFrame:
    """Summary of development metrics per typology cluster."""
    df = df.copy()
    df["cluster"]  = labels
    df["typology"] = [cluster_to_typology.get(l, f"Type_{l}") for l in labels]

    cols = ["development_index", "revitalization_need",
            "growth_potential", "economic_vitality",
            "social_welfare", "infrastructure_score"]
    cols = [c for c in cols if c in df.columns]

    summary = df.groupby("typology")[cols].agg(["mean", "std"]).round(3)
    tier_counts = df.groupby(["typology", "priority_tier"]).size().unstack(fill_value=0)

    return summary, tier_counts
