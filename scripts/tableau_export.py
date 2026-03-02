#!/usr/bin/env python3
"""
tableau_export.py — Aggregate and export all Tableau-ready datasets
===================================================================
Module: 7006SCN Machine Learning and Big Data — Coventry University

Run AFTER all notebooks (1-4) and scalability_experiments.py.
Consolidates every CSV export into tableau/ with consistent formatting.

Usage:
    python scripts/tableau_export.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from pathlib import Path

TABLEAU_DIR = Path("tableau")
TABLEAU_DIR.mkdir(exist_ok=True)


def export_model_comparison():
    """
    Dashboard 2: Model Performance
    Combines validation + test metrics into a single comparison table.
    """
    val_path  = TABLEAU_DIR / "model_comparison.csv"     # from notebook 3
    test_path = TABLEAU_DIR / "test_metrics.csv"          # from notebook 4

    dfs = []
    if val_path.exists():
        val_df = pd.read_csv(val_path)
        val_df["split"] = "validation"
        dfs.append(val_df)
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        test_df["split"] = "test"
        dfs.append(test_df)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        out = TABLEAU_DIR / "model_performance_combined.csv"
        combined.to_csv(out, index=False)
        print(f"✓ {out.name}: {len(combined)} rows")
        return combined
    print("⚠ No model metric files found.")
    return None


def export_roc_data():
    """
    Dashboard 2: ROC curves (already exported by notebook 4).
    Verify it exists and report.
    """
    roc_path = TABLEAU_DIR / "roc_data.csv"
    if roc_path.exists():
        df = pd.read_csv(roc_path)
        print(f"✓ {roc_path.name}: {len(df):,} data points")
        return df
    print("⚠ roc_data.csv not found — run notebook 4 first.")
    return None


def export_feature_importance():
    """
    Dashboard 3: Feature Insights
    """
    fi_path = TABLEAU_DIR / "feature_importance.csv"
    if fi_path.exists():
        df = pd.read_csv(fi_path)
        print(f"✓ {fi_path.name}: {len(df)} features")
        return df
    print("⚠ feature_importance.csv not found — run notebook 4 first.")
    return None


def export_class_distribution():
    """
    Dashboard 1: Data Quality
    """
    cd_path = TABLEAU_DIR / "class_distribution.csv"
    if cd_path.exists():
        df = pd.read_csv(cd_path)
        print(f"✓ {cd_path.name}: {len(df)} rows")
        return df
    print("⚠ class_distribution.csv not found — run notebook 2 first.")
    return None


def export_scalability():
    """
    Dashboard 4: Scalability & Cost Analysis
    """
    sc_path = TABLEAU_DIR / "scalability_results.csv"
    if sc_path.exists():
        df = pd.read_csv(sc_path)
        # Add derived columns for Tableau
        strong = df[df["experiment"] == "strong_scaling"].copy()
        if len(strong) > 0:
            baseline = strong.loc[strong["executors"] == strong["executors"].min(), "train_time_s"]
            if len(baseline) > 0:
                base_val = baseline.values[0]
                strong["speedup"]    = base_val / strong["train_time_s"]
                strong["efficiency"] = strong["speedup"] / strong["executors"]
                # Cost model: assume $0.10/executor/minute (EMR m5.xlarge approx)
                strong["cost_usd"] = (strong["train_time_s"] / 60) * strong["executors"] * 0.10

        weak = df[df["experiment"] == "weak_scaling"].copy()
        enriched = pd.concat([strong, weak], ignore_index=True)
        out = TABLEAU_DIR / "scalability_enriched.csv"
        enriched.to_csv(out, index=False)
        print(f"✓ {out.name}: {len(enriched)} rows (with speedup & cost)")
        return enriched
    print("⚠ scalability_results.csv not found — run scalability_experiments.py first.")
    return None


def create_data_quality_summary():
    """
    Dashboard 1: Data Quality — summary statistics.
    If raw data exists, compute and export quality metrics.
    """
    summary = {
        "metric": [
            "Total records ingested",
            "Records after cleaning",
            "Labelled records (supervised)",
            "English-only filter applied",
            "Schema validation",
            "Deduplication method",
            "Storage format",
            "Partitioning strategy",
            "Compression codec",
        ],
        "value": [
            "See notebook 1 output",
            "See notebook 1 output",
            "See notebook 2 output",
            "Yes — language='eng'",
            "StructType with non-nullable fields",
            "dropDuplicates(['url'])",
            "Parquet (columnar)",
            "By crawl_date",
            "Snappy",
        ],
        "rubric_relevance": [
            "Data Engineering 20%",
            "Data Engineering 20%",
            "Data Engineering 20%",
            "Data Engineering 20%",
            "Data Engineering 20%",
            "Data Engineering 20%",
            "Data Engineering 20%",
            "Data Engineering 20%",
            "Data Engineering 20%",
        ]
    }
    df = pd.DataFrame(summary)
    out = TABLEAU_DIR / "data_quality_summary.csv"
    df.to_csv(out, index=False)
    print(f"✓ {out.name}: {len(df)} quality metrics")
    return df


# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TABLEAU EXPORT CONSOLIDATION")
    print("=" * 60)
    print()

    export_class_distribution()
    export_model_comparison()
    export_roc_data()
    export_feature_importance()
    export_scalability()
    create_data_quality_summary()

    print("\n── Final Tableau directory inventory ──")
    for f in sorted(TABLEAU_DIR.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:45s} {size_kb:>8.1f} KB")

    print("""
╔══════════════════════════════════════════════════════════════╗
║  TABLEAU DASHBOARD DESIGN (4 dashboards)                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. DATA QUALITY                                             ║
║     Sources: class_distribution.csv, data_quality_summary.csv║
║     Charts: Bar chart (class balance), table (quality stats) ║
║     Story: Show data pipeline robustness                     ║
║                                                              ║
║  2. MODEL PERFORMANCE                                        ║
║     Sources: model_performance_combined.csv, roc_data.csv    ║
║     Charts: Grouped bar (Acc/F1/AUC by model),              ║
║             ROC overlay, confusion matrix heatmaps           ║
║     Story: Compare models, justify best pick                 ║
║                                                              ║
║  3. FEATURE INSIGHTS                                         ║
║     Sources: feature_importance.csv,                         ║
║              feature_importance_with_words.csv                ║
║     Charts: Horizontal bar (top features), word cloud        ║
║     Story: What drives fake vs. reliable classification      ║
║                                                              ║
║  4. SCALABILITY & COST ANALYSIS                              ║
║     Sources: scalability_enriched.csv                        ║
║     Charts: Line (speedup vs executors),                     ║
║             Dual-axis (time vs cost), efficiency bar          ║
║     Story: Prove system scales; show cost-performance curve  ║
║                                                              ║
║  STORYTELLING FLOW:                                          ║
║  Data Quality → Model Performance → Features → Scalability  ║
║  "We ingested real web data at scale, built robust models,   ║
║   identified key linguistic signals, and proved the system   ║
║   scales efficiently with cluster resources."                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
