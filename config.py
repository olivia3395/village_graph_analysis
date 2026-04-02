"""
config.py — Configuration for Village Graph Analysis.

Project context
───────────────
Rural revitalization planning in Ji County (蓟县), Tianjin, China.
The system builds a graph over villages where nodes represent individual
villages and edges encode spatial proximity and socio-economic similarity.
Graph learning and clustering identify village typologies and development
clusters to support evidence-based planning decisions.

Multi-source indicators (node features)
─────────────────────────────────────────
  Geographic  : location, elevation, slope, distance to county seat
  Economic    : per-capita income, employment rate, industry type
  Social      : population, age structure, education level, healthcare access
  Infrastructure: road quality, broadband coverage, utility access
  Agricultural: cultivated land area, crop diversity, mechanisation rate
  Planning    : current land-use category, development stage, policy zone
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DataConfig:
    """Village data settings."""
    data_dir: str = "data/raw"
    output_dir: str = "outputs"

    # Synthetic data generation
    n_villages: int = 200          # number of villages to simulate
    n_features: int = 24           # number of socio-economic indicators
    seed: int = 42

    # Feature groups and their column ranges
    feature_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "geographic":     ["longitude", "latitude", "elevation_m",
                           "slope_deg", "dist_to_county_km"],
        "economic":       ["income_per_capita_cny", "employment_rate",
                           "industry_diversity", "tourism_revenue_cny",
                           "market_access_score"],
        "social":         ["population", "youth_ratio", "elderly_ratio",
                           "education_index", "healthcare_access"],
        "infrastructure": ["road_quality", "broadband_coverage",
                           "water_supply", "electricity_reliability"],
        "agricultural":   ["cultivated_area_mu", "crop_diversity",
                           "mechanisation_rate", "irrigation_coverage"],
        "planning":       ["development_stage", "policy_priority",
                           "land_use_intensity", "environmental_constraint"],
    })


@dataclass
class GraphConfig:
    """Graph construction settings."""

    # Edge construction strategy
    # "spatial"    : connect villages within distance_threshold_km
    # "knn"        : k-nearest neighbours by feature similarity
    # "hybrid"     : weighted combination of spatial + feature similarity
    edge_strategy: str = "hybrid"

    # Spatial threshold for edge creation (km)
    distance_threshold_km: float = 15.0

    # k for kNN graph
    knn_k: int = 8

    # Weight of spatial vs feature similarity in hybrid mode
    spatial_weight: float = 0.4
    feature_weight: float = 0.6

    # Minimum edge weight to keep (prune weak edges)
    min_edge_weight: float = 0.1

    # Self-loops
    add_self_loops: bool = False


@dataclass
class ClusteringConfig:
    """Clustering model settings."""

    # Number of village typology clusters
    n_clusters: int = 5

    # Spectral clustering
    spectral_affinity: str = "precomputed"   # use the adjacency matrix
    spectral_assign_labels: str = "kmeans"
    spectral_n_init: int = 20
    spectral_random_state: int = 42

    # Louvain community detection
    louvain_resolution: float = 1.0          # higher → more communities
    louvain_seed: int = 42

    # Node2Vec embedding (for graph-based clustering)
    n2v_dimensions: int = 32
    n2v_walk_length: int = 20
    n2v_num_walks: int = 100
    n2v_p: float = 1.0                       # return parameter
    n2v_q: float = 0.5                       # in-out parameter (DFS-like)
    n2v_window: int = 5
    n2v_epochs: int = 5

    # Ensemble: majority vote across methods
    use_ensemble: bool = True


@dataclass
class TypologyConfig:
    """Village typology label definitions."""

    # Canonical typology labels for Ji County context
    typology_names: Dict[int, str] = field(default_factory=lambda: {
        0: "Agricultural Core",          # high cultivated area, mechanised farming
        1: "Tourism & Cultural",         # scenic resources, heritage sites
        2: "Peri-urban Transition",      # high road quality, proximity to county seat
        3: "Underdeveloped Priority",    # low income, elderly-heavy, needs investment
        4: "Industrial & Commercial",    # diverse industry, market access
    })

    # Development priority scores per typology (for revitalization targeting)
    development_priority: Dict[int, float] = field(default_factory=lambda: {
        0: 0.5,   # moderate — stable but needs modernisation
        1: 0.7,   # high — tourism potential underutilised
        2: 0.4,   # lower — already benefits from urban spillover
        3: 1.0,   # highest — direct revitalization target
        4: 0.3,   # lower — relatively self-sufficient
    })


@dataclass
class Config:
    data:      DataConfig      = field(default_factory=DataConfig)
    graph:     GraphConfig     = field(default_factory=GraphConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    typology:  TypologyConfig  = field(default_factory=TypologyConfig)

    def summary(self) -> str:
        lines = ["=" * 56, "  Village Graph Analysis — Configuration", "=" * 56]
        for sec in ("data", "graph", "clustering"):
            obj = getattr(self, sec)
            lines.append(f"\n[{sec.upper()}]")
            for k, v in vars(obj).items():
                if not isinstance(v, dict):
                    lines.append(f"  {k:<30} {v}")
        lines.append("=" * 56)
        return "\n".join(lines)
