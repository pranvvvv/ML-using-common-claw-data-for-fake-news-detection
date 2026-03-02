"""
test_spark_session.py — Verify SparkSession factory works correctly.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_spark_session_creates():
    """SparkSession should start with config from YAML."""
    from config.spark_session import get_spark, stop_spark
    spark = get_spark()
    assert spark is not None
    assert spark.version.startswith("3.")
    stop_spark(spark)


def test_spark_config_loaded():
    """Key config values from spark_config.yaml should be applied."""
    from config.spark_session import get_spark, stop_spark
    spark = get_spark()
    conf = spark.sparkContext.getConf()

    # Arrow optimization must be enabled
    assert conf.get("spark.sql.execution.arrow.pyspark.enabled") == "true"
    # AQE must be on
    assert conf.get("spark.sql.adaptive.enabled") == "true"
    # Anonymous S3 for Common Crawl
    assert "Anonymous" in conf.get("spark.hadoop.fs.s3a.aws.credentials.provider", "")

    stop_spark(spark)


def test_scaling_profile_override():
    """Scaling profiles should override executor config."""
    from config.spark_session import get_spark, stop_spark
    spark = get_spark(profile="single")
    conf = spark.sparkContext.getConf()
    assert conf.get("spark.executor.instances") == "1"
    stop_spark(spark)
