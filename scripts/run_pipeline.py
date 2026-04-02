"""
scripts/run_pipeline.py — Full village graph analysis pipeline.

Usage:
  python scripts/run_pipeline.py                    # synthetic data
  python scripts/run_pipeline.py --data path/to.csv # real CSV
  python scripts/run_pipeline.py --no-plots         # skip visualisation
"""

import argparse, os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from config import Config
from data.loader import load_data
from data.preprocessing import preprocess
from graph.builder import VillageGraphBuilder
from graph.features import extract_graph_features
from models.spectral import SpectralClusterer
from models.community import LouvainDetector
from models.graph_embedding import Node2VecEmbedder
from analysis.typology import label_clusters, typology_profile
from analysis.development import compute_development_scores, rank_villages, cluster_development_summary
from evaluation.metrics import internal_metrics, external_metrics, print_metrics_table


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default=None,   help="Path to village CSV")
    p.add_argument("--n-villages", type=int, default=200)
    p.add_argument("--n-clusters", type=int, default=5)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--no-plots",   action="store_true")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()
    cfg.data.n_villages         = args.n_villages
    cfg.data.seed               = args.seed
    cfg.clustering.n_clusters   = args.n_clusters
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*58}")
    print("  Village Graph Analysis — Ji County Rural Revitalization")
    print(f"{'='*58}")
    print(cfg.summary())

    # 1. Load data
    df, true_labels = load_data(args.data, args.n_villages, args.seed)

    # 2. Preprocess
    df_clean, X, feat_names = preprocess(df, cfg.data.feature_groups)
    village_ids = list(df_clean.index)
    print(f"  Feature matrix: {X.shape}")

    # 3. Build graph
    builder = VillageGraphBuilder(cfg.graph)
    G = builder.build(df_clean, X, village_ids)

    # 4. Extract graph structural features
    print("\n[Graph Features]")
    gf = extract_graph_features(G)
    for col in gf.columns:
        df_clean[col] = gf[col]

    # 5. Run all clustering methods
    print("\n[Clustering]")
    sc      = SpectralClusterer(cfg.clustering)
    louvain = LouvainDetector(cfg.clustering)
    n2v     = Node2VecEmbedder(cfg.clustering)

    labels_spectral  = sc.fit_predict(G, village_ids)
    labels_louvain   = louvain.fit_predict(G, village_ids)
    labels_n2v       = n2v.fit_predict(G, village_ids)

    # 6. Evaluate
    print("\n[Evaluation]")
    all_metrics = {}
    for method, labels in [("Spectral", labels_spectral),
                            ("Louvain",  labels_louvain),
                            ("Node2Vec", labels_n2v)]:
        m = internal_metrics(X, labels)
        if true_labels is not None:
            m.update(external_metrics(labels, true_labels))
        all_metrics[method] = m

    print_metrics_table(all_metrics, "Clustering Method Comparison")

    # Use spectral labels as primary result
    labels_primary = labels_spectral

    # 7. Typology labelling
    cluster_to_typology = label_clusters(df_clean, labels_primary, village_ids, cfg.typology)
    print("\n  Cluster → Typology mapping:")
    for cid, tname in cluster_to_typology.items():
        n = (labels_primary == cid).sum()
        print(f"    Cluster {cid} ({n:>3} villages) → {tname}")

    profile = typology_profile(df_clean, labels_primary, cluster_to_typology)
    print(f"\n  Typology feature profile:\n{profile.to_string()}")

    # 8. Development scoring
    df_clean = compute_development_scores(df_clean)
    df_clean["cluster"]  = labels_primary
    df_clean["typology"] = [cluster_to_typology.get(l, f"Type_{l}") for l in labels_primary]

    top20 = rank_villages(df_clean, by="revitalization_need", top_n=20)
    print(f"\n  Top 20 villages by revitalization need:\n{top20.to_string()}")

    dev_summary, tier_counts = cluster_development_summary(df_clean, labels_primary, cluster_to_typology)
    print(f"\n  Development summary by typology:\n{dev_summary.to_string()}")
    print(f"\n  Priority tier counts:\n{tier_counts.to_string()}")

    # 9. Save results
    df_clean.to_csv(os.path.join(args.output_dir, "village_results.csv"))
    profile.to_csv(os.path.join(args.output_dir, "typology_profile.csv"))
    top20.to_csv(os.path.join(args.output_dir, "top20_priority.csv"))
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n  Results saved → {args.output_dir}/")

    # 10. Plots
    if not args.no_plots:
        from visualization.plots import (
            plot_spatial_clusters, plot_graph,
            plot_typology_radar, plot_development_heatmap,
            plot_metrics_comparison,
        )
        od = args.output_dir
        plot_spatial_clusters(df_clean, labels_primary, cluster_to_typology,
                               save_path=f"{od}/spatial_clusters.png", show=False)
        plot_graph(G, labels_primary, village_ids, cluster_to_typology,
                   save_path=f"{od}/village_graph.png", show=False)
        plot_typology_radar(profile,
                            save_path=f"{od}/typology_radar.png", show=False)
        plot_development_heatmap(df_clean, "development_index",
                                  save_path=f"{od}/development_heatmap.png", show=False)
        plot_metrics_comparison(all_metrics,
                                 save_path=f"{od}/method_comparison.png", show=False)
        print(f"  Plots saved → {od}/")

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
