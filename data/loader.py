"""
data/loader.py — Village data loader (real CSV or synthetic fallback).

For the Ji County project, the real data came from:
  - County Planning Bureau GIS exports (spatial + land use)
  - National Rural Statistical Yearbook (socio-economic)
  - China Rural Household Survey (income, employment)
  - Tianjin Infrastructure Database (roads, utilities)

This loader handles both real CSV data and the synthetic fallback.
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from data.synthetic import generate_village_data


def load_data(
    data_path: Optional[str] = None,
    n_villages: int = 200,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Load village data from a CSV file or generate synthetic data.

    Args:
        data_path  : path to a CSV file with village indicators.
                     Columns must include 'longitude', 'latitude',
                     and at least some of the indicator columns.
                     If None → use synthetic data.
        n_villages : number of villages for synthetic mode
        seed       : random seed

    Returns:
        df          : DataFrame indexed by village_id
        true_labels : ground-truth typology labels if available, else None
    """
    if data_path and os.path.exists(data_path):
        print(f"[Data] Loading from: {data_path}")
        df = pd.read_csv(data_path, index_col=0)
        true_labels = df.pop("true_typology").values if "true_typology" in df.columns else None
        print(f"  {len(df)} villages, {len(df.columns)} features")
        return df, true_labels

    print(f"[Data] Generating synthetic data ({n_villages} villages)...")
    df, true_labels = generate_village_data(n_villages=n_villages, seed=seed)
    print(f"  {len(df)} villages, {len(df.columns)} features")
    print(f"  True typology distribution: "
          f"{dict(zip(*np.unique(true_labels, return_counts=True)))}")
    return df, true_labels
