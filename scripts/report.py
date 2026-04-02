"""
scripts/report.py — Generate a structured planning report from analysis results.

Usage:
  python scripts/report.py
  python scripts/report.py --output-dir outputs/report
"""

import argparse, os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-json", default="outputs/results.json")
    p.add_argument("--output-dir",   default="outputs/report")
    return p.parse_args()


def generate_text_report(results: dict, output_dir: str):
    """Write a human-readable planning report to a .txt file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "village_planning_report.txt")

    lines = [
        "=" * 64,
        "  VILLAGE TYPOLOGY & DEVELOPMENT ANALYSIS REPORT",
        "  Ji County Rural Revitalization Planning Project",
        "=" * 64,
        "",
        f"  Villages analysed    : {results.get('n_villages', 'N/A')}",
        f"  Features used        : {results.get('n_features', 'N/A')}",
        f"  Graph edges          : {results.get('n_edges', 'N/A')}",
        f"  Clusters identified  : {results.get('n_clusters', 'N/A')}",
        f"  Best method (ARI)    : {results.get('best_method', 'N/A')}",
        "",
        "─" * 64,
        "  CLUSTER QUALITY METRICS",
        "─" * 64,
    ]

    for method, metrics in results.get("metrics", {}).items():
        lines.append(f"\n  {method}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"    {k:<25} {v:.4f}")
            else:
                lines.append(f"    {k:<25} {v}")

    lines += [
        "",
        "─" * 64,
        "  VILLAGE TYPOLOGY PROFILES",
        "─" * 64,
    ]
    for tname, profile in results.get("typology_profiles", {}).items():
        lines.append(f"\n  {tname}:")
        for feat, val in profile.items():
            lines.append(f"    {feat:<30} {val:.3f}")

    lines += [
        "",
        "─" * 64,
        "  TOP PRIORITY VILLAGES (Revitalization Need)",
        "─" * 64,
    ]
    for row in results.get("priority_villages", [])[:15]:
        lines.append(
            f"  {row.get('village_id','?'):<10}  "
            f"need={row.get('revitalization_need', 0):.3f}  "
            f"{row.get('priority_tier','')}"
        )

    lines += ["", "=" * 64, "  End of Report", "=" * 64]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report saved → {path}")


def main():
    args = parse_args()

    # Load results or run pipeline inline
    if os.path.exists(args.results_json):
        with open(args.results_json) as f:
            results = json.load(f)
        print(f"Loaded results from {args.results_json}")
    else:
        print("No results.json found — running pipeline first...")
        from scripts.run_pipeline import run_pipeline
        results = run_pipeline(save_plots=False, verbose=False)

    generate_text_report(results, args.output_dir)
    print("Report generation complete.")


if __name__ == "__main__":
    main()
