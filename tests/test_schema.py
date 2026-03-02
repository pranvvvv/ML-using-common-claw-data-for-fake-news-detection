"""
test_schema.py — Verify the article schema and data validation logic.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.sql.types import StructType, StructField, StringType, IntegerType


ARTICLE_SCHEMA = StructType([
    StructField("url",            StringType(),  nullable=False),
    StructField("crawl_date",     StringType(),  nullable=False),
    StructField("content_length", IntegerType(), nullable=True),
    StructField("language",       StringType(),  nullable=True),
    StructField("text",           StringType(),  nullable=False),
])


def test_schema_field_count():
    """Schema should have exactly 5 fields."""
    assert len(ARTICLE_SCHEMA.fields) == 5


def test_schema_non_nullable_fields():
    """url, crawl_date, text must be non-nullable."""
    non_nullable = {f.name for f in ARTICLE_SCHEMA.fields if not f.nullable}
    assert non_nullable == {"url", "crawl_date", "text"}


def test_schema_creates_valid_dataframe():
    """Schema should accept valid records."""
    from config.spark_session import get_spark, stop_spark
    spark = get_spark()

    data = [
        ("https://example.com", "2024-03-01", 1500, "eng", "This is a test article with sufficient length for processing and analysis."),
    ]
    df = spark.createDataFrame(data, schema=ARTICLE_SCHEMA)
    assert df.count() == 1
    assert df.columns == ["url", "crawl_date", "content_length", "language", "text"]

    stop_spark(spark)


def test_schema_rejects_wrong_types():
    """Schema should reject data with wrong column types."""
    from config.spark_session import get_spark, stop_spark
    spark = get_spark()

    # content_length should be IntegerType, not a string that can't be cast
    data = [("https://example.com", "2024-03-01", "not_a_number", "eng", "text")]
    with pytest.raises(Exception):
        spark.createDataFrame(data, schema=ARTICLE_SCHEMA).collect()

    stop_spark(spark)
