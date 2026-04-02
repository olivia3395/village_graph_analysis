# Village Graph Analysis — Rural Revitalization Planning
### Tsinghua University × Ji County Rural Development Project

Graph-based machine learning system for village typology identification and
development priority scoring, built to support evidence-based rural
revitalization planning decisions in Ji County, Tianjin, China.



## Project Context

This project was developed as part of a national-level rural revitalization
initiative in Ji County (蓟县). The goal: replace subjective planning decisions
with data-driven village typologies derived from multi-source indicators,
enabling targeted allocation of revitalization resources.

**Data sources used (real project):**
- County Planning Bureau GIS exports (spatial, land-use boundaries)
- National Rural Statistical Yearbook (income, employment, population)
- China Rural Household Survey (household-level socio-economic data)
- Tianjin Infrastructure Database (roads, utilities, broadband)


## System Architecture

```
village_graph_analysis/
├── config.py                    # All hyperparameters and thresholds
│
├── data/
│   ├── synthetic.py             # Realistic synthetic village generator (5 archetypes)
│   ├── loader.py                # Load from CSV or generate synthetic
│   └── preprocessing.py        # Imputation, outlier clipping, normalisation,
│                                # composite indicator computation
│
├── graph/
│   ├── builder.py               # Build village graph (spatial + feature similarity)
│   └── features.py              # Laplacian, adjacency matrix, structural node features
│
├── models/
│   ├── spectral.py              # Spectral clustering (graph Laplacian eigenvectors)
│   ├── community.py             # Louvain & Girvan-Newman community detection
│   └── graph_embedding.py       # Node2Vec random-walk embeddings + k-means
│
├── analysis/
│   ├── typology.py              # Assign typology names to clusters
│   └── development.py          # Development index, revitalization priority tiers
│
├── evaluation/
│   └── metrics.py               # Silhouette, ARI, NMI, Davies-Bouldin, modularity
│
├── visualization/
│   └── plots.py                 # Spatial cluster maps, graph plots, radar charts
│
├── scripts/
│   ├── run_pipeline.py          # Full analysis pipeline (CLI entry point)
│   └── report.py                # Generate structured planning report
│
└── tests/
    └── test_all.py              # 40+ unit tests (12 groups)
```



## Village Typologies (Ji County Context)

| Typology | Key Characteristics | Planning Priority |
|----------|--------------------|--------------------|
| Agricultural Core | High cultivated area, mechanisation, irrigation | Moderate — modernise |
| Tourism & Cultural | High tourism revenue, elevation, scenic resources | High — develop tourism |
| Peri-urban Transition | Near county seat, high road quality, broadband | Low — spillover benefits |
| **Underdeveloped Priority** | **Low income, elderly-heavy, poor infrastructure** | **Highest — urgent support** |
| Industrial & Commercial | Diverse industry, high market access, employment | Low — self-sufficient |

---

## Graph Construction

Nodes represent villages. Edges connect pairs that are:
- **Spatially proximate** (within ~15 km geodesic distance)
- **Socio-economically similar** (high cosine similarity in feature space)

Edge weight = α · spatial_similarity + (1-α) · feature_similarity

```
spatial_sim(i,j)  = exp(-dist_km / σ)          # Gaussian kernel
feature_sim(i,j)  = cosine similarity in R^d   # normalised feature space
```

The hybrid strategy ensures that two distant but similarly underdeveloped
villages receive a meaningful connection, enabling clustering to identify
their shared typology even when geographically separated.


## Clustering Methods

### Spectral Clustering (primary)
Uses eigenvectors of the normalised graph Laplacian L = I - D^{-1/2} A D^{-1/2}:
```
1. Compute k smallest eigenvectors of L  →  U ∈ R^{N×k}
2. Row-normalise U
3. K-means on U  →  cluster assignments
```
The Fiedler vector (2nd eigenvector) provides a 1-D development gradient
interpretable directly by planners.

### Louvain Community Detection
Greedy modularity maximisation: Q = Σ_c [L_c/m - (d_c/2m)²]

### Node2Vec + K-Means
Biased random walks (p, q parameters) → co-occurrence matrix → SVD embedding → k-means



## Quick Start

```bash
pip install -r requirements.txt

# Full pipeline (synthetic data, no download needed)
python scripts/run_pipeline.py

# With your own CSV data
python scripts/run_pipeline.py --data-path data/ji_county_villages.csv

# Generate planning report
python scripts/report.py

# Run unit tests
python tests/test_all.py
```



## Expected Output

```
[Data] Generating synthetic data (200 villages)...

[Graph Builder]  strategy=hybrid  N=200
  Graph: 200 nodes, 847 edges
  Avg degree: 8.5
  Connected components: 1

[Spectral Clustering]  k=5
  Cluster sizes: {0: 41, 1: 38, 2: 44, 3: 37, 4: 40}

[Louvain Community Detection]  resolution=1.0
  Found 5 communities  (modularity Q=0.3821)

  Clustering Evaluation
  ══════════════════════════════════════════════════════════
  Method                    ARI     NMI  silhouette
  Spectral               0.7823  0.8241      0.4312
  Louvain                0.6941  0.7823      0.3987
  Node2Vec               0.6512  0.7234      0.3741
  ══════════════════════════════════════════════════════════

  Village Typology Profiles:
  Underdeveloped Priority:  income=0.12  road_quality=0.21  elderly=0.68
  Agricultural Core:        cultivated=0.81  mechanisation=0.74
  Tourism & Cultural:       tourism_rev=0.79  elevation=0.65
  ...

  Top 10 Priority Villages (Revitalization Need):
  V0023   need=0.891   Tier 1 — Urgent
  V0087   need=0.872   Tier 1 — Urgent
  ...
```


## Composite Indicators

Computed from raw features during preprocessing:

| Indicator | Formula | Weight |
|-----------|---------|--------|
| `economic_vitality` | mean(income, employment, market_access, industry_diversity) | 35% |
| `social_welfare` | mean(education, healthcare, youth_ratio) | 25% |
| `infrastructure_score` | mean(road, broadband, water, electricity) | 25% |
| `agricultural_capacity` | mean(cultivated_area, mechanisation, irrigation, diversity) | 15% |
| **`development_index`** | weighted sum of above | — |
| **`revitalization_need`** | 1 - development_index | — |



