###############################################################################
# spark_session.py — Single factory for every notebook / script
# Reads config/spark_config.yaml  →  returns a tuned SparkSession.
###############################################################################
import os, yaml, findspark
from pathlib import Path
from pyspark.sql import SparkSession

findspark.init()  # locate SPARK_HOME automatically

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH  = _PROJECT_ROOT / "config" / "spark_config.yaml"


def get_spark(profile: str | None = None) -> SparkSession:
    """
    Build (or retrieve) a SparkSession with production-grade defaults.

    Parameters
    ----------
    profile : str, optional
        One of the scaling_profiles keys in spark_config.yaml
        ('single', 'dual', 'quad').  If None, uses top-level defaults.

    Returns
    -------
    SparkSession
    """
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    builder = SparkSession.builder.appName(cfg["app_name"])

    # Apply base config
    for k, v in cfg["spark_config"].items():
        builder = builder.config(k, v)

    # Override with scaling profile if requested
    if profile and profile in cfg.get("scaling_profiles", {}):
        for k, v in cfg["scaling_profiles"][profile].items():
            builder = builder.config(k, v)

    spark = builder.getOrCreate()

    # Set log level to WARN to reduce notebook noise
    spark.sparkContext.setLogLevel("WARN")

    return spark


def stop_spark(spark: SparkSession) -> None:
    """Gracefully stop the session and release resources."""
    spark.stop()
