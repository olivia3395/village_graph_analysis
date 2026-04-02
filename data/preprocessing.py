"""
data/preprocessing.py — Multi-source data cleaning and normalisation.

Steps
─────
  1. Missing value imputation (median per feature group)
  2. Outlier clipping (IQR-based, per feature)
  3. Min-max normalisation per feature group
  4. Optional: PCA dimensionality reduction per group
  5. Composite indicator computation (development index, etc.)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Median imputation per numeric column."""
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df


# ---------------------------------------------------------------------------
# Outlier clipping
# ---------------------------------------------------------------------------

def clip_outliers(df: pd.DataFrame, iqr_factor: float = 3.0) -> pd.DataFrame:
    """Clip values beyond median ± iqr_factor × IQR."""
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - iqr_factor * iqr
        hi  = q3 + iqr_factor * iqr
        df[col] = df[col].clip(lo, hi)
    return df


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def minmax_normalise(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Min-max normalise selected columns to [0, 1]."""
    df   = df.copy()
    cols = feature_cols or df.select_dtypes(include=[np.number]).columns.tolist()
    for col in cols:
        mn, mx = df[col].min(), df[col].max()
        if mx > mn:
            df[col] = (df[col] - mn) / (mx - mn)
        else:
            df[col] = 0.0
    return df


def zscore_normalise(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Z-score standardise selected columns."""
    df   = df.copy()
    cols = feature_cols or df.select_dtypes(include=[np.number]).columns.tolist()
    for col in cols:
        mu, sd = df[col].mean(), df[col].std()
        df[col] = (df[col] - mu) / max(sd, 1e-9)
    return df


# ---------------------------------------------------------------------------
# Composite indicators
# ---------------------------------------------------------------------------

def compute_composite_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute higher-level composite scores from raw indicators.

    Added columns:
        economic_vitality    : weighted average of income, employment, market access
        social_welfare       : education + healthcare + youth retention
        infrastructure_score : road + broadband + water + electricity
        agricultural_capacity: land + mechanisation + irrigation + crop diversity
        development_index    : overall development level (0=low, 1=high)
        revitalization_need  : inverse of development_index (priority for support)
    """
    df = df.copy()

    def _safe_mean(cols):
        available = [c for c in cols if c in df.columns]
        return df[available].mean(axis=1) if available else 0.0

    df["economic_vitality"] = _safe_mean([
        "income_per_capita_cny", "employment_rate",
        "market_access_score", "industry_diversity",
    ])
    df["social_welfare"] = _safe_mean([
        "education_index", "healthcare_access",
        "youth_ratio",
    ])
    df["infrastructure_score"] = _safe_mean([
        "road_quality", "broadband_coverage",
        "water_supply", "electricity_reliability",
    ])
    df["agricultural_capacity"] = _safe_mean([
        "cultivated_area_mu", "mechanisation_rate",
        "irrigation_coverage", "crop_diversity",
    ])

    # Overall development index
    df["development_index"] = (
        0.35 * df["economic_vitality"] +
        0.25 * df["social_welfare"] +
        0.25 * df["infrastructure_score"] +
        0.15 * df["agricultural_capacity"]
    )
    df["revitalization_need"] = 1.0 - df["development_index"]

    return df


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    normalise: str = "minmax",
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Full preprocessing pipeline.

    Args:
        df             : raw village DataFrame (index = village_id)
        feature_groups : dict of group_name → list of column names
        normalise      : "minmax" | "zscore"

    Returns:
        df_clean   : cleaned DataFrame with composite indicators
        X          : (n_villages, n_features) float32 feature matrix
        feat_names : list of feature column names
    """
    print("[Preprocessing]")

    # 1. Impute
    df = impute_missing(df)
    print(f"  Imputation done")

    # 2. Clip outliers
    df = clip_outliers(df)
    print(f"  Outlier clipping done")

    # 3. Compute composites (before normalisation — uses raw scales)
    df = compute_composite_indicators(df)

    # 4. Collect feature columns
    feat_cols = []
    for group, cols in feature_groups.items():
        for c in cols:
            if c in df.columns:
                feat_cols.append(c)
    # Add composite indicators
    for ci in ("economic_vitality", "social_welfare",
               "infrastructure_score", "agricultural_capacity",
               "development_index"):
        if ci in df.columns and ci not in feat_cols:
            feat_cols.append(ci)

    # 5. Normalise
    if normalise == "minmax":
        df = minmax_normalise(df, feat_cols)
    else:
        df = zscore_normalise(df, feat_cols)
    print(f"  Normalisation ({normalise}) done — {len(feat_cols)} features")

    X = df[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    return df, X, feat_cols
