"""
test_pipeline.py — Verify the NLP feature pipeline logic.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_tokenizer_splits_text():
    """Tokenizer should split text into words."""
    from config.spark_session import get_spark, stop_spark
    from pyspark.ml.feature import Tokenizer

    spark = get_spark()
    df = spark.createDataFrame([("hello world foo bar",)], ["text"])
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    result = tokenizer.transform(df).select("words").collect()[0].words
    assert result == ["hello", "world", "foo", "bar"]
    stop_spark(spark)


def test_stopwords_remover():
    """StopWordsRemover should remove English stop words."""
    from config.spark_session import get_spark, stop_spark
    from pyspark.ml.feature import StopWordsRemover

    spark = get_spark()
    df = spark.createDataFrame([(["the", "cat", "is", "on", "the", "mat"],)], ["words"])
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    result = remover.transform(df).select("filtered").collect()[0].filtered
    # "the", "is", "on" are stop words
    assert "the" not in result
    assert "cat" in result
    assert "mat" in result
    stop_spark(spark)


def test_hashing_tf_output_dimension():
    """HashingTF should produce vectors of specified dimension."""
    from config.spark_session import get_spark, stop_spark
    from pyspark.ml.feature import HashingTF

    spark = get_spark()
    num_features = 1024
    df = spark.createDataFrame([(["hello", "world"],)], ["words"])
    htf = HashingTF(inputCol="words", outputCol="features", numFeatures=num_features)
    result = htf.transform(df).select("features").collect()[0].features
    assert result.size == num_features
    stop_spark(spark)


def test_full_pipeline_produces_features():
    """End-to-end: text → Tokenizer → StopWords → HTF → IDF → feature vector."""
    from config.spark_session import get_spark, stop_spark
    from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
    from pyspark.ml import Pipeline

    spark = get_spark()
    data = [
        (0, "the quick brown fox jumps over the lazy dog"),
        (1, "spark is a distributed computing framework for big data"),
        (0, "machine learning on big data requires distributed algorithms"),
    ]
    df = spark.createDataFrame(data, ["label", "text"])

    pipeline = Pipeline(stages=[
        Tokenizer(inputCol="text", outputCol="words"),
        StopWordsRemover(inputCol="words", outputCol="filtered"),
        HashingTF(inputCol="filtered", outputCol="raw_features", numFeatures=256),
        IDF(inputCol="raw_features", outputCol="features", minDocFreq=1),
    ])

    model = pipeline.fit(df)
    result = model.transform(df)

    assert "features" in result.columns
    assert result.count() == 3
    # Features should be non-zero vectors
    first_vec = result.select("features").collect()[0].features
    assert first_vec.numNonzeros() > 0

    stop_spark(spark)
