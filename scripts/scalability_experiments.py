#!/usr/bin/env python3
"""
scalability_experiments.py — Strong & Weak Scaling Benchmarks
=============================================================
Module: 7006SCN Machine Learning and Big Data — Coventry University
Project: Scalable Fake News Detection on Common Crawl

Usage:
    python scripts/scalability_experiments.py

This script:
1. Runs the SAME training workload with 1, 2, and 4 executor configs (strong scaling).
2. Runs PROPORTIONALLY increasing data with proportionally increasing executors (weak scaling).
3. Collects: training time, shuffle read/write, executor utilisation.
4. Exports results to tableau/scalability_results.csv.
"""

import sys, os, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from pathlib import Path
from config.spark_session import get_spark, stop_spark

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark import StorageLevel


# ─── Configuration ──────────────────────────────────────────────────────────
FEATURES_TRAIN = "data/parquet/features/train"
TABLEAU_DIR    = Path("tableau")
TABLEAU_DIR.mkdir(exist_ok=True)

PROFILES = ["single", "dual", "quad"]   # maps to scaling_profiles in spark_config.yaml
EXECUTOR_COUNTS = {"single": 1, "dual": 2, "quad": 4}


# ─── Helper: collect Spark metrics ─────────────────────────────────────────
def collect_stage_metrics(spark):
    """
    Parse the Spark REST API (localhost:4040) for the most recent job's
    stage-level shuffle metrics.

    Returns dict: {shuffle_read_bytes, shuffle_write_bytes, total_task_time_ms}
    """
    try:
        import requests
        base = spark.sparkContext.uiWebUrl
        apps = requests.get(f"{base}/api/v1/applications").json()
        app_id = apps[0]["id"]
        stages = requests.get(f"{base}/api/v1/applications/{app_id}/stages").json()

        total_shuffle_read  = sum(s.get("shuffleReadBytes", 0) for s in stages)
        total_shuffle_write = sum(s.get("shuffleWriteBytes", 0) for s in stages)
        total_task_time     = sum(s.get("executorRunTime", 0) for s in stages)

        return {
            "shuffle_read_MB":   round(total_shuffle_read / 1e6, 2),
            "shuffle_write_MB":  round(total_shuffle_write / 1e6, 2),
            "total_task_time_s": round(total_task_time / 1000, 2),
        }
    except Exception as e:
        print(f"  ⚠ Could not collect Spark metrics: {e}")
        return {"shuffle_read_MB": None, "shuffle_write_MB": None, "total_task_time_s": None}


# ─── Strong Scaling Experiment ──────────────────────────────────────────────
def strong_scaling():
    """
    STRONG SCALING: Fix the dataset size, vary executors (1 → 2 → 4).
    Ideal speedup = N (linear); actual speedup shows parallelism efficiency.
    """
    print("\n" + "=" * 70)
    print("STRONG SCALING EXPERIMENT")
    print("Fixed data size, increasing executor count")
    print("=" * 70)

    results = []

    for profile in PROFILES:
        print(f"\n── Profile: {profile} ({EXECUTOR_COUNTS[profile]} executors) ──")

        # Fresh SparkSession with this profile's config
        spark = get_spark(profile=profile)

        train_df = spark.read.parquet(FEATURES_TRAIN)
        train_df.persist(StorageLevel.MEMORY_AND_DISK)
        row_count = train_df.count()
        print(f"  Data size: {row_count:,} rows")

        # Train Logistic Regression (deterministic workload)
        lr = LogisticRegression(
            featuresCol="features", labelCol="label",
            maxIter=50, regParam=0.1, elasticNetParam=0.0
        )

        t0 = time.time()
        model = lr.fit(train_df)
        train_time = time.time() - t0

        metrics = collect_stage_metrics(spark)

        results.append({
            "experiment":    "strong_scaling",
            "profile":       profile,
            "executors":     EXECUTOR_COUNTS[profile],
            "data_rows":     row_count,
            "train_time_s":  round(train_time, 2),
            **metrics,
        })

        print(f"  Train time: {train_time:.2f}s")
        print(f"  Shuffle read: {metrics['shuffle_read_MB']} MB")

        train_df.unpersist()
        stop_spark(spark)

    return results


# ─── Weak Scaling Experiment ────────────────────────────────────────────────
def weak_scaling():
    """
    WEAK SCALING: Increase data AND executors proportionally.
    If infrastructure scales perfectly, training time stays constant.
    """
    print("\n" + "=" * 70)
    print("WEAK SCALING EXPERIMENT")
    print("Data size and executors scale together")
    print("=" * 70)

    results = []
    fractions = {"single": 0.25, "dual": 0.5, "quad": 1.0}

    for profile in PROFILES:
        frac = fractions[profile]
        print(f"\n── Profile: {profile} | Data fraction: {frac:.0%} ──")

        spark = get_spark(profile=profile)

        full_df = spark.read.parquet(FEATURES_TRAIN)

        # Sample proportional to executor count
        train_df = full_df.sample(fraction=frac, seed=42)
        train_df.persist(StorageLevel.MEMORY_AND_DISK)
        row_count = train_df.count()
        print(f"  Data size: {row_count:,} rows  |  Executors: {EXECUTOR_COUNTS[profile]}")

        lr = LogisticRegression(
            featuresCol="features", labelCol="label",
            maxIter=50, regParam=0.1, elasticNetParam=0.0
        )

        t0 = time.time()
        model = lr.fit(train_df)
        train_time = time.time() - t0

        metrics = collect_stage_metrics(spark)

        results.append({
            "experiment":    "weak_scaling",
            "profile":       profile,
            "executors":     EXECUTOR_COUNTS[profile],
            "data_fraction": frac,
            "data_rows":     row_count,
            "train_time_s":  round(train_time, 2),
            **metrics,
        })

        print(f"  Train time: {train_time:.2f}s")

        train_df.unpersist()
        stop_spark(spark)

    return results


# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    strong_results = strong_scaling()
    weak_results   = weak_scaling()

    all_results = strong_results + weak_results
    df = pd.DataFrame(all_results)

    output_path = TABLEAU_DIR / "scalability_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Exported scalability results to {output_path}")
    print(df.to_string(index=False))

    # ── Compute speedup & efficiency ──
    print("\n── Strong Scaling Analysis ──")
    strong_df = df[df["experiment"] == "strong_scaling"].copy()
    baseline_time = strong_df.loc[strong_df["executors"] == 1, "train_time_s"].values[0]
    strong_df["speedup"]    = baseline_time / strong_df["train_time_s"]
    strong_df["efficiency"] = strong_df["speedup"] / strong_df["executors"]
    print(strong_df[["executors", "train_time_s", "speedup", "efficiency"]].to_string(index=False))

    print("\n── Weak Scaling Analysis ──")
    weak_df = df[df["experiment"] == "weak_scaling"].copy()
    print(weak_df[["executors", "data_rows", "train_time_s"]].to_string(index=False))
    print("  Ideal: training time stays constant as both data and executors scale.")
