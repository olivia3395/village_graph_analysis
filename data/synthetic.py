"""
data/synthetic.py — Synthetic village dataset generator for Ji County.

Generates realistic multi-source socio-economic indicators for N villages,
with spatial coordinates placed on a realistic county-level grid and
feature correlations that reflect actual rural development patterns.

Five latent village archetypes are embedded in the data:
  0 - Agricultural Core      : high land area, mechanisation, crop diversity
  1 - Tourism & Cultural     : high tourism revenue, heritage score, scenic
  2 - Peri-urban Transition  : high road quality, low dist_to_county, broadband
  3 - Underdeveloped Priority: low income, high elderly ratio, poor infrastructure
  4 - Industrial & Commercial: high income, industry diversity, market access

These correspond to the five clusters the analysis should recover.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple


# Archetype centres in feature space (24 features)
# Order: lon, lat, elev, slope, dist_county,
#        income, employment, industry_div, tourism_rev, market_access,
#        population, youth_ratio, elderly_ratio, education, healthcare,
#        road_quality, broadband, water, electricity,
#        cultivated_area, crop_diversity, mechanisation, irrigation,
#        dev_stage, policy_priority, land_use, env_constraint
#        (last 4 = planning block, 24 total)

ARCHETYPES = {
    0: {  # Agricultural Core
        "income_per_capita_cny": 18000, "employment_rate": 0.72,
        "industry_diversity": 0.30,     "tourism_revenue_cny": 500,
        "market_access_score": 0.45,    "population": 600,
        "youth_ratio": 0.28,            "elderly_ratio": 0.18,
        "education_index": 0.55,        "healthcare_access": 0.60,
        "road_quality": 0.60,           "broadband_coverage": 0.55,
        "cultivated_area_mu": 1800,     "crop_diversity": 0.65,
        "mechanisation_rate": 0.75,     "irrigation_coverage": 0.70,
        "elevation_m": 120,             "dist_to_county_km": 18,
    },
    1: {  # Tourism & Cultural
        "income_per_capita_cny": 24000, "employment_rate": 0.68,
        "industry_diversity": 0.55,     "tourism_revenue_cny": 85000,
        "market_access_score": 0.60,    "population": 450,
        "youth_ratio": 0.32,            "elderly_ratio": 0.15,
        "education_index": 0.62,        "healthcare_access": 0.65,
        "road_quality": 0.75,           "broadband_coverage": 0.72,
        "cultivated_area_mu": 600,      "crop_diversity": 0.45,
        "mechanisation_rate": 0.40,     "irrigation_coverage": 0.45,
        "elevation_m": 280,             "dist_to_county_km": 22,
    },
    2: {  # Peri-urban Transition
        "income_per_capita_cny": 32000, "employment_rate": 0.80,
        "industry_diversity": 0.70,     "tourism_revenue_cny": 2000,
        "market_access_score": 0.85,    "population": 1200,
        "youth_ratio": 0.35,            "elderly_ratio": 0.12,
        "education_index": 0.75,        "healthcare_access": 0.82,
        "road_quality": 0.90,           "broadband_coverage": 0.90,
        "cultivated_area_mu": 400,      "crop_diversity": 0.30,
        "mechanisation_rate": 0.55,     "irrigation_coverage": 0.60,
        "elevation_m": 60,              "dist_to_county_km": 6,
    },
    3: {  # Underdeveloped Priority
        "income_per_capita_cny": 9000,  "employment_rate": 0.50,
        "industry_diversity": 0.15,     "tourism_revenue_cny": 100,
        "market_access_score": 0.20,    "population": 280,
        "youth_ratio": 0.18,            "elderly_ratio": 0.35,
        "education_index": 0.35,        "healthcare_access": 0.30,
        "road_quality": 0.30,           "broadband_coverage": 0.25,
        "cultivated_area_mu": 900,      "crop_diversity": 0.35,
        "mechanisation_rate": 0.30,     "irrigation_coverage": 0.35,
        "elevation_m": 350,             "dist_to_county_km": 35,
    },
    4: {  # Industrial & Commercial
        "income_per_capita_cny": 42000, "employment_rate": 0.85,
        "industry_diversity": 0.85,     "tourism_revenue_cny": 3000,
        "market_access_score": 0.90,    "population": 2000,
        "youth_ratio": 0.38,            "elderly_ratio": 0.10,
        "education_index": 0.80,        "healthcare_access": 0.88,
        "road_quality": 0.92,           "broadband_coverage": 0.95,
        "cultivated_area_mu": 200,      "crop_diversity": 0.20,
        "mechanisation_rate": 0.50,     "irrigation_coverage": 0.50,
        "elevation_m": 45,              "dist_to_county_km": 4,
    },
}

# Spatial cluster centres (longitude, latitude) — Ji County area
SPATIAL_CENTRES = {
    0: (117.55, 40.00),   # Agricultural — inland plains
    1: (117.65, 40.12),   # Tourism — northern hills
    2: (117.45, 39.90),   # Peri-urban — near county seat
    3: (117.72, 40.18),   # Underdeveloped — remote hills
    4: (117.42, 39.88),   # Industrial — southern corridor
}

FEATURE_STDS = {
    "income_per_capita_cny": 4000, "employment_rate": 0.08,
    "industry_diversity": 0.12,    "tourism_revenue_cny": 8000,
    "market_access_score": 0.10,   "population": 200,
    "youth_ratio": 0.05,           "elderly_ratio": 0.05,
    "education_index": 0.08,       "healthcare_access": 0.08,
    "road_quality": 0.10,          "broadband_coverage": 0.10,
    "cultivated_area_mu": 250,     "crop_diversity": 0.10,
    "mechanisation_rate": 0.10,    "irrigation_coverage": 0.10,
    "elevation_m": 40,             "dist_to_county_km": 4,
}


def generate_village_data(
    n_villages: int = 200,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic village socio-economic data with spatial coordinates.

    Returns:
        df          : DataFrame with n_villages rows and all feature columns
        true_labels : (n_villages,) integer array of ground-truth typology
    """
    rng = np.random.RandomState(seed)

    # Assign each village to an archetype
    n_per_type = n_villages // 5
    remainder  = n_villages % 5
    counts     = [n_per_type + (1 if i < remainder else 0) for i in range(5)]
    labels     = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
    rng.shuffle(labels)

    rows = []
    for vid, tc in enumerate(labels):
        arch    = ARCHETYPES[tc]
        sp_ctr  = SPATIAL_CENTRES[tc]

        # Spatial position with cluster spread
        lon = sp_ctr[0] + rng.normal(0, 0.06)
        lat = sp_ctr[1] + rng.normal(0, 0.06)

        row = {
            "village_id":  f"V{vid:04d}",
            "longitude":   round(lon, 5),
            "latitude":    round(lat, 5),
            "true_typology": int(tc),
        }

        for feat, centre in arch.items():
            std = FEATURE_STDS.get(feat, centre * 0.10)
            val = rng.normal(centre, std)
            # Clip to plausible range
            if "ratio" in feat or "rate" in feat or "coverage" in feat \
               or "quality" in feat or "index" in feat or "score" in feat \
               or "intensity" in feat or "constraint" in feat or "diversity" in feat:
                val = float(np.clip(val, 0.0, 1.0))
            elif feat in ("elevation_m", "dist_to_county_km",
                          "population", "cultivated_area_mu",
                          "income_per_capita_cny", "tourism_revenue_cny"):
                val = max(0.0, val)
            row[feat] = round(val, 3)

        # Remaining planning fields not in ARCHETYPES
        row["slope_deg"]            = round(abs(rng.normal(10 + tc * 5, 4)), 1)
        row["development_stage"]    = round(float(np.clip(rng.normal(0.3 + tc * 0.1, 0.1), 0, 1)), 3)
        row["policy_priority"]      = round(float(np.clip(rng.normal(0.5 - tc * 0.05, 0.15), 0, 1)), 3)
        row["land_use_intensity"]   = round(float(np.clip(rng.normal(0.4 + tc * 0.08, 0.1), 0, 1)), 3)
        row["environmental_constraint"] = round(float(np.clip(rng.normal(0.3 - tc * 0.03, 0.1), 0, 1)), 3)

        rows.append(row)

    df = pd.DataFrame(rows).set_index("village_id")
    true_labels = df.pop("true_typology").values
    return df, true_labels
