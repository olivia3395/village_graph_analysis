<div align="center">

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/NetworkX-Graph_ML-e11d48?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Tsinghua_University-782F9E?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Ji_County_Tianjin-Rural_Revitalization-16a34a?style=for-the-badge"/>

<br/><br/>

# 🏘️ Village Graph Analysis
### Rural Revitalization Planning · Ji County, Tianjin, China

<br/>

> **Tsinghua University × Ji County Rural Development Project**  
> Graph-based ML system for village typology identification and development priority scoring —  
> replacing subjective planning decisions with evidence-based, data-driven analysis.

<br/>

[🚀 Quick Start](#-quick-start) · [🗺️ Village Typologies](#️-village-typologies) · [🕸️ Graph Construction](#️-graph-construction) · [📊 Expected Output](#-expected-output) · [🏗️ Architecture](#️-system-architecture)

<br/>



</div>

## 📌 Project Context

This project was developed as part of a **national-level rural revitalization initiative** in Ji County (蓟县). The goal: replace subjective planning decisions with data-driven village typologies derived from multi-source indicators, enabling targeted allocation of revitalization resources.

<br/>

**Real data sources used:**

| Source | Contents |
|:---|:---|
| 🗂️ County Planning Bureau GIS exports | Spatial boundaries, land-use data |
| 📊 National Rural Statistical Yearbook | Income, employment, population |
| 🏠 China Rural Household Survey | Household-level socio-economic data |
| 🛣️ Tianjin Infrastructure Database | Roads, utilities, broadband coverage |

<br/>



## 🗺️ Village Typologies

Five archetypes identified for Ji County, each with distinct planning priorities:

<br/>

<div align="center">

| Typology | Key Characteristics | Planning Priority |
|:---|:---|:---:|
| 🌾 **Agricultural Core** | High cultivated area, mechanisation, irrigation | Moderate — modernise |
| 🏔️ **Tourism & Cultural** | High tourism revenue, elevation, scenic resources | High — develop tourism |
| 🏙️ **Peri-urban Transition** | Near county seat, high road quality, broadband | Low — spillover benefits |
| 🚨 **Underdeveloped Priority** | Low income, elderly-heavy, poor infrastructure | **Highest — urgent support** |
| 🏭 **Industrial & Commercial** | Diverse industry, high market access, employment | Low — self-sufficient |

</div>

<br/>



## 🕸️ Graph Construction

Villages are represented as nodes; edges connect pairs that are:

- 📍 **Spatially proximate** — within ~15 km geodesic distance
- 📐 **Socio-economically similar** — high cosine similarity in feature space

```
Edge weight = α · spatial_similarity + (1-α) · feature_similarity

  spatial_sim(i,j)  =  exp(-dist_km / σ)           Gaussian kernel
  feature_sim(i,j)  =  cosine similarity in R^d     normalised feature space
```

> The hybrid strategy ensures that two distant but similarly underdeveloped villages receive a meaningful connection — enabling clustering to identify their shared typology even when geographically separated.

<br/>



## 🧮 Clustering Methods

Three complementary algorithms are run and compared:

<br/>

**Spectral Clustering** *(primary)*

Computes eigenvectors of the normalised graph Laplacian, then applies k-means in the embedding space. The Fiedler vector (2nd eigenvector) provides a 1-D development gradient directly interpretable by planners.

```
1. Compute k smallest eigenvectors of L  →  U ∈ R^{N×k}
2. Row-normalise U
3. K-means on U  →  cluster assignments
```

**Louvain Community Detection**

Greedy modularity maximisation over the village graph. Produces communities that respect both geographic and socio-economic proximity.

**Node2Vec + K-Means**

Biased random walks on the graph generate node embeddings, which are then clustered with k-means. The `p` and `q` parameters control the trade-off between local and global graph structure.

<br/>



## 📐 Composite Indicators

Computed from raw features during preprocessing:

<div align="center">

| Indicator | Components | Weight |
|:---|:---|:---:|
| `economic_vitality` | income · employment · market access · industry diversity | **35%** |
| `social_welfare` | education · healthcare · youth ratio | **25%** |
| `infrastructure_score` | road · broadband · water · electricity | **25%** |
| `agricultural_capacity` | cultivated area · mechanisation · irrigation · diversity | **15%** |
| **`development_index`** | weighted sum of above | — |
| **`revitalization_need`** | `1 − development_index` | — |

</div>

<br/>



## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run full pipeline — synthetic data, no download needed

```bash
python scripts/run_pipeline.py
```

### Run with your own CSV data

```bash
python scripts/run_pipeline.py --data-path data/ji_county_villages.csv
```

### Generate planning report

```bash
python scripts/report.py
```

### Run unit tests

```bash
python tests/test_all.py
```

<br/>



## 📊 Expected Output

```
[Data] Generating synthetic data (200 villages) ...

[Graph Builder]  strategy=hybrid  N=200
  Graph: 200 nodes · 847 edges
  Avg degree: 8.5  |  Connected components: 1

[Spectral Clustering]  k=5
  Cluster sizes: {0: 41, 1: 38, 2: 44, 3: 37, 4: 40}

[Louvain Community Detection]  resolution=1.0
  Found 5 communities  (modularity Q=0.3821)

  Clustering Evaluation
  ══════════════════════════════════════════════════════════
  Method                    ARI     NMI  Silhouette
  Spectral               0.7823  0.8241      0.4312
  Louvain                0.6941  0.7823      0.3987
  Node2Vec               0.6512  0.7234      0.3741
  ══════════════════════════════════════════════════════════

  Village Typology Profiles
  Underdeveloped Priority:  income=0.12  road_quality=0.21  elderly=0.68
  Agricultural Core:        cultivated=0.81  mechanisation=0.74
  Tourism & Cultural:       tourism_rev=0.79  elevation=0.65

  Top 10 Priority Villages (Revitalization Need)
  V0023   need=0.891   🔴 Tier 1 — Urgent
  V0087   need=0.872   🔴 Tier 1 — Urgent
  ...
```

<br/>



## 🏗️ System Architecture

```
village_graph_analysis/
│
├── config.py                    All hyperparameters and thresholds
│
├── data/
│   ├── synthetic.py             Realistic synthetic village generator  (5 archetypes)
│   ├── loader.py                Load from CSV or generate synthetic data
│   └── preprocessing.py        Imputation · outlier clipping · normalisation
│                                composite indicator computation
│
├── graph/
│   ├── builder.py               Build village graph  (spatial + feature similarity)
│   └── features.py              Laplacian · adjacency matrix · structural node features
│
├── models/
│   ├── spectral.py              Spectral clustering  (graph Laplacian eigenvectors)
│   ├── community.py             Louvain & Girvan-Newman community detection
│   └── graph_embedding.py       Node2Vec random-walk embeddings + k-means
│
├── analysis/
│   ├── typology.py              Assign typology names to clusters
│   └── development.py           Development index · revitalization priority tiers
│
├── evaluation/
│   └── metrics.py               Silhouette · ARI · NMI · Davies-Bouldin · modularity
│
├── visualization/
│   └── plots.py                 Spatial cluster maps · graph plots · radar charts
│
├── scripts/
│   ├── run_pipeline.py          Full analysis pipeline  (CLI entry point)
│   └── report.py                Generate structured planning report
│
└── tests/
    └── test_all.py              40+ unit tests across 12 groups
```

<br/>


## 🧪 Testing

```bash
python -m pytest tests/ -v
# or
python tests/test_all.py
```

**40+ tests across 12 groups** — covering data generation, preprocessing, graph construction, spectral clustering, community detection, Node2Vec embeddings, typology assignment, development scoring, evaluation metrics, visualization, pipeline integration, and report generation.

<br/>



<div align="center">

*Data-driven planning for the villages that need it most.*

</div>
