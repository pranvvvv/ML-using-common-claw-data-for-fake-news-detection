"""
Scalable Fake News Detection — Full Pipeline Runner
=====================================================
Module: 7006SCN Machine Learning and Big Data — Coventry University
Runs Notebooks 1→4 logic + scalability + Tableau export end-to-end.
"""
import os, sys, time, json, pickle, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

warnings.filterwarnings("ignore")

# ─── Environment setup ──────────────────────────────────────────────────
os.environ["JAVA_HOME"]              = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot"
os.environ["HADOOP_HOME"]            = r"C:\hadoop"
os.environ["PATH"]                   = r"C:\hadoop\bin;" + os.environ.get("PATH", "")
os.environ["PYSPARK_PYTHON"]         = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"]  = sys.executable

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import (
    LogisticRegression, LinearSVC, RandomForestClassifier,
    LogisticRegressionModel, LinearSVCModel, RandomForestClassificationModel,
)
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark import StorageLevel

# ─── Paths ───────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_RAW    = ROOT / "data" / "raw"
DATA_PARQUET= ROOT / "data" / "parquet"
FEATURES    = DATA_PARQUET / "features"
MODELS_DIR  = ROOT / "data" / "models"
TABLEAU_DIR = ROOT / "tableau"

for d in [DATA_RAW, DATA_PARQUET, FEATURES, MODELS_DIR, TABLEAU_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Spark Session ───────────────────────────────────────────────────────
print("=" * 70)
print("  CREATING SPARK SESSION")
print("=" * 70)
spark = (
    SparkSession.builder
    .appName("FakeNewsDetection")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.ui.showConsoleProgress", "false")
    .getOrCreate()
)
sc = spark.sparkContext
sc.setLogLevel("ERROR")
print(f"Spark version : {spark.version}")
print(f"Spark UI      : {sc.uiWebUrl}")

# =====================================================================
# NOTEBOOK 1 — Data Ingestion & Preparation
# =====================================================================
print("\n" + "=" * 70)
print("  NOTEBOOK 1 — DATA INGESTION")
print("=" * 70)
t_nb1 = time.time()

np.random.seed(42)

reliable_sources   = ["reuters.com","apnews.com","bbc.com","nytimes.com","washingtonpost.com","theguardian.com","npr.org"]
unreliable_sources = ["infowars.com","naturalnews.com","beforeitsnews.com","worldnewsdailyreport.com","newspunch.com"]
subjects_real = ["politicsNews","worldnews","business","science","technology"]
subjects_fake = ["News","politics","Government News","left-news","US_News"]

real_phrases = [
    "According to official reports released today, {topic} experts confirmed that recent developments in {area} have led to significant policy changes. The data suggests a consistent trend toward {direction}, with analysts noting historical patterns support this. Government officials stated new measures will address ongoing challenges in {sector}.",
    "A comprehensive study published in a peer-reviewed journal found that {topic} has shown measurable improvements over the past decade. Researchers analyzed data from multiple sources and concluded evidence-based approaches yield positive results in {area} and {sector}.",
    "Officials announced today that {topic} legislation has been passed following bipartisan negotiations. The bill addresses key concerns in {area} and {sector}, with provisions for increased funding. Economic analysts predict moderate positive impact on {direction}.",
    "International diplomats gathered to discuss {topic} challenges facing the global community. Representatives from over 40 countries produced a joint statement on {area} commitments with specific timelines for {sector} improvements.",
    "The quarterly economic report shows {topic} indicators trending {direction} for the third consecutive period. Labor market data reveals steady progress in {area}, with financial markets responding to {sector} developments.",
]
fake_phrases = [
    "BREAKING: Shocking revelations about {topic} that mainstream media REFUSES to report! A massive conspiracy involving {area} goes all the way to the top. Share before they DELETE it! Millions deceived about {sector}.",
    "You wont believe what they discovered about {topic}!!! Secret LEAKED documents show everything about {area} is a lie. The establishment hides the truth about {sector}. Wake up people!",
    "EXPOSED: Hidden truth about {topic} THEY dont want you to see. Anonymous insiders expose {area} cover-ups. This changes everything about {sector}. Evidence is UNDENIABLE.",
    "URGENT: New evidence proves {topic} scandal far WORSE than imagined. Deep state manipulating {area} for decades. Patriots must fight lies about {sector}. Forward to everyone!",
    "BOMBSHELL: What they HIDE about {topic} will SHOCK you. Whistleblowers expose {area} fraud at unprecedented scale. Cover-up reaches highest levels of {sector}. Media BLACKOUT proves complicity.",
]
topics     = ["healthcare","economy","climate","education","immigration","technology","defense","trade","energy","infrastructure"]
areas      = ["policy","reform","regulation","research","development","spending","cooperation","planning","assessment","governance"]
sectors    = ["public health","financial markets","environmental protection","national security","international relations","digital privacy"]
directions = ["improvement","growth","stabilization","recovery","expansion"]
dates      = pd.date_range("2017-01-01","2018-12-31",freq="D").strftime("%B %d, %Y").tolist()

def gen(templates, sources, subjects, n):
    rows = []
    for i in range(n):
        t1, t2 = np.random.choice(templates, 2, replace=True)
        kw = lambda: dict(topic=np.random.choice(topics), area=np.random.choice(areas),
                          sector=np.random.choice(sectors), direction=np.random.choice(directions))
        rows.append({
            "title": f"Article-{np.random.choice(topics)}-{i}",
            "text": t1.format(**kw()) + " " + t2.format(**kw()),
            "subject": np.random.choice(subjects),
            "date": np.random.choice(dates),
            "source": np.random.choice(sources),
        })
    return pd.DataFrame(rows)

true_pd = gen(real_phrases, reliable_sources, subjects_real, 21000)
fake_pd = gen(fake_phrases, unreliable_sources, subjects_fake, 23000)
true_pd.to_csv(str(DATA_RAW / "True.csv"), index=False)
fake_pd.to_csv(str(DATA_RAW / "Fake.csv"), index=False)
print(f"Generated — Real: {len(true_pd):,} | Fake: {len(fake_pd):,} | Total: {len(true_pd)+len(fake_pd):,}")

# Combine & label
true_pd["label"] = 0
fake_pd["label"] = 1
combined_pd = pd.concat([true_pd, fake_pd], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"Combined (shuffled): {len(combined_pd):,} articles")

# Schema enforcement → Spark DataFrame (via Arrow)
ARTICLE_SCHEMA = StructType([
    StructField("title",   StringType(),  nullable=True),
    StructField("text",    StringType(),  nullable=False),
    StructField("subject", StringType(),  nullable=True),
    StructField("date",    StringType(),  nullable=True),
    StructField("source",  StringType(),  nullable=True),
    StructField("label",   IntegerType(), nullable=False),
])
raw_df = spark.createDataFrame(combined_pd, schema=ARTICLE_SCHEMA)
print(f"Spark DF: {raw_df.count():,} rows | Partitions: {raw_df.rdd.getNumPartitions()}")
raw_df.printSchema()

# Handle missing/corrupt
print("=== Null / Empty audit ===")
for c in raw_df.columns:
    n = raw_df.filter(F.col(c).isNull() | (F.trim(F.col(c).cast("string")) == "")).count()
    print(f"  {c:15s} -> {n:>6,} nulls/empty")

clean_df = (raw_df
    .dropna(subset=["text"])
    .filter(F.length("text") >= 100)
    .dropDuplicates(["text"])
    .fillna({"title":"Unknown","subject":"general","date":"unknown","source":"unknown"})
)
print(f"\nBefore: {raw_df.count():>10,}")
print(f"After:  {clean_df.count():>10,}")

# Persist + write Parquet
clean_df.persist(StorageLevel.MEMORY_AND_DISK)
row_count = clean_df.count()
print(f"Cached {row_count:,} rows")

OUTPUT_PATH = str(DATA_PARQUET / "news_articles")
clean_df.repartition("subject").write.mode("overwrite").partitionBy("subject").parquet(OUTPUT_PATH)
print(f"Parquet written -> {OUTPUT_PATH}")

verify_df = spark.read.parquet(OUTPUT_PATH)
print(f"  Verified: {verify_df.count():,} rows | Parts: {verify_df.rdd.getNumPartitions()} | Subjects: {verify_df.select('subject').distinct().count()}")

clean_df.unpersist()
summary_nb1 = verify_df.groupBy("label").count().toPandas()
summary_nb1["label_name"] = summary_nb1["label"].map({0: "Reliable", 1: "Fake"})
print(summary_nb1.to_string(index=False))
print(f"\nNotebook 1 complete — {row_count:,} articles -> Parquet ({time.time()-t_nb1:.1f}s)")

# =====================================================================
# NOTEBOOK 2 — Feature Engineering
# =====================================================================
print("\n" + "=" * 70)
print("  NOTEBOOK 2 — FEATURE ENGINEERING")
print("=" * 70)
t_nb2 = time.time()

# Load Parquet
df = spark.read.parquet(OUTPUT_PATH)
print(f"Loaded {df.count():,} rows | Partitions: {df.rdd.getNumPartitions()}")

# Text preprocessing (using built-in Spark SQL functions, no Python UDFs)
from pyspark.sql.functions import regexp_replace, trim, length, lower, col

df = (
    df
    .withColumn("text", regexp_replace("text", r"https?://\S+", ""))
    .withColumn("text", regexp_replace("text", r"[^a-zA-Z\s]", ""))
    .withColumn("text", regexp_replace("text", r"\s+", " "))
    .withColumn("text", trim(lower(col("text"))))
    .filter(length("text") >= 100)
)
print(f"After text cleaning: {df.count():,} rows")

# MLlib Feature Pipeline
tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=2**14)
idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=5)

feature_pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf])

print("Feature pipeline stages:")
for i, stage in enumerate(feature_pipeline.getStages()):
    print(f"  {i}: {type(stage).__name__}")

# Fit + Transform
df.persist(StorageLevel.MEMORY_AND_DISK)
pipeline_model = feature_pipeline.fit(df)
features_df = pipeline_model.transform(df)
features_df = features_df.select("title", "source", "subject", "label", "features")
features_df.persist(StorageLevel.MEMORY_AND_DISK)
feat_count = features_df.count()
print(f"Feature vectors: {feat_count:,} rows")

df.unpersist()

# Save pipeline model
pipeline_model.write().overwrite().save(str(MODELS_DIR / "feature_pipeline"))
print("Feature pipeline model saved")

# Stratified train/val/test split
fractions_train = {0: 0.7, 1: 0.7}
train_df = features_df.stat.sampleBy("label", fractions_train, seed=42)
remaining = features_df.subtract(train_df)

fractions_vt = {0: 0.5, 1: 0.5}
val_df  = remaining.stat.sampleBy("label", fractions_vt, seed=42)
test_df = remaining.subtract(val_df)

for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    total = split_df.count()
    pos   = split_df.filter(col("label") == 1).count()
    print(f"{name:6s}: {total:>8,} rows  |  label=1 ratio: {pos/max(total,1):.3f}")

# Save splits
for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    path = str(FEATURES / name)
    split_df.write.mode("overwrite").parquet(path)
    print(f"  {name} -> {path}")

# Class distribution for Tableau
class_dist = features_df.groupBy("label").agg(F.count("*").alias("count")).toPandas()
class_dist["label_name"] = class_dist["label"].map({0: "Reliable", 1: "Unreliable"})
class_dist.to_csv(str(TABLEAU_DIR / "class_distribution.csv"), index=False)
print(f"Exported class_distribution.csv for Tableau")

features_df.unpersist()
print(f"\nNotebook 2 complete ({time.time()-t_nb2:.1f}s)")

# =====================================================================
# NOTEBOOK 3 — Model Training
# =====================================================================
print("\n" + "=" * 70)
print("  NOTEBOOK 3 — MODEL TRAINING")
print("=" * 70)
t_nb3 = time.time()

# Reload splits
train_df = spark.read.parquet(str(FEATURES / "train")).persist(StorageLevel.MEMORY_AND_DISK)
val_df   = spark.read.parquet(str(FEATURES / "val")).persist(StorageLevel.MEMORY_AND_DISK)
test_df  = spark.read.parquet(str(FEATURES / "test")).persist(StorageLevel.MEMORY_AND_DISK)
print(f"Train: {train_df.count():>8,}  |  Val: {val_df.count():>6,}  |  Test: {test_df.count():>6,}")

# Model definitions
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100, family="binomial")
lr_grid = (ParamGridBuilder()
    .addGrid(lr.regParam, [0.01, 0.1])
    .addGrid(lr.elasticNetParam, [0.0, 0.5])
    .build())

svc = LinearSVC(featuresCol="features", labelCol="label", maxIter=100)
svc_grid = (ParamGridBuilder()
    .addGrid(svc.regParam, [0.01, 0.1])
    .build())

print(f"LR grid : {len(lr_grid)} | SVC grid: {len(svc_grid)}")

auc_evaluator = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)

# --- Train LR and SVC with 3-fold CV ---
cv_models_config = {
    "LogisticRegression": (lr, lr_grid),
    "LinearSVC":          (svc, svc_grid),
}

NUM_FOLDS = 3
cv_results = {}

for name, (estimator, grid) in cv_models_config.items():
    print(f"\n{'='*60}")
    print(f"Training: {name}  |  Grid combos: {len(grid)}  |  Folds: {NUM_FOLDS}")
    print(f"{'='*60}")

    cv = CrossValidator(
        estimator=estimator,
        estimatorParamMaps=grid,
        evaluator=auc_evaluator,
        numFolds=NUM_FOLDS,
        parallelism=2,
        seed=42,
    )

    t0 = time.time()
    cv_model = cv.fit(train_df)
    train_time = time.time() - t0

    best_model = cv_model.bestModel
    avg_metrics = cv_model.avgMetrics
    best_auc = max(avg_metrics)

    cv_results[name] = {
        "best_model": best_model,
        "cv_model": cv_model,
        "best_auc": best_auc,
        "train_time": train_time,
        "avg_metrics": avg_metrics,
    }

    print(f"  Best CV AUC  : {best_auc:.4f}")
    print(f"  Train time   : {train_time:.1f}s")

    best_idx = avg_metrics.index(best_auc)
    best_params = grid[best_idx]
    for param, val in best_params.items():
        print(f"  Best {param.name}: {val}")

# --- Train RandomForest directly (no CV — too memory-intensive for high-dim TF-IDF) ---
print(f"\n{'='*60}")
print(f"Training: RandomForest  |  Direct fit (numTrees=100, maxDepth=5)")
print(f"{'='*60}")

rf = RandomForestClassifier(
    featuresCol="features", labelCol="label",
    numTrees=100, maxDepth=5, maxBins=32, seed=42,
)
t0 = time.time()
rf_model = rf.fit(train_df)
rf_train_time = time.time() - t0

# Evaluate RF on train set for AUC
rf_train_preds = rf_model.transform(train_df)
rf_train_auc = auc_evaluator.evaluate(rf_train_preds)

cv_results["RandomForest"] = {
    "best_model": rf_model,
    "cv_model": None,
    "best_auc": rf_train_auc,
    "train_time": rf_train_time,
    "avg_metrics": [rf_train_auc],
}
print(f"  Train AUC    : {rf_train_auc:.4f}")
print(f"  Train time   : {rf_train_time:.1f}s")

# Validation evaluation
acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_eval  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

val_metrics = []
for name, res in cv_results.items():
    preds = res["best_model"].transform(val_df)
    accuracy = acc_eval.evaluate(preds)
    f1 = f1_eval.evaluate(preds)
    try:
        auc_val = auc_evaluator.evaluate(preds)
    except Exception:
        auc_val = None

    val_metrics.append({
        "model": name,
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "auc": round(auc_val, 4) if auc_val else "N/A",
        "train_time_s": round(res["train_time"], 1),
    })
    print(f"{name:25s}  Acc={accuracy:.4f}  F1={f1:.4f}  AUC={auc_val if auc_val else 'N/A'}")

val_metrics_df = pd.DataFrame(val_metrics)
print("\n", val_metrics_df.to_string(index=False))

# Save models
for name, res in cv_results.items():
    mllib_path = str(MODELS_DIR / f"{name}_mllib")
    res["best_model"].write().overwrite().save(mllib_path)
    print(f"  {name} -> {mllib_path}")

    pkl_path = MODELS_DIR / f"{name}_cvmodel.pkl"
    meta = {"model_name": name, "best_auc": res["best_auc"],
            "train_time": res["train_time"], "avg_metrics": res["avg_metrics"]}
    with open(pkl_path, "wb") as fh:
        pickle.dump(meta, fh)

val_metrics_df.to_csv(str(TABLEAU_DIR / "model_comparison.csv"), index=False)
print("Exported model_comparison.csv")

train_df.unpersist(); val_df.unpersist(); test_df.unpersist()
print(f"\nNotebook 3 complete ({time.time()-t_nb3:.1f}s)")

# =====================================================================
# NOTEBOOK 4 — Evaluation
# =====================================================================
print("\n" + "=" * 70)
print("  NOTEBOOK 4 — EVALUATION")
print("=" * 70)
t_nb4 = time.time()

test_df = spark.read.parquet(str(FEATURES / "test")).persist(StorageLevel.MEMORY_AND_DISK)
print(f"Test set: {test_df.count():,} rows")

models = {
    "LogisticRegression": LogisticRegressionModel.load(str(MODELS_DIR / "LogisticRegression_mllib")),
    "LinearSVC":          LinearSVCModel.load(str(MODELS_DIR / "LinearSVC_mllib")),
    "RandomForest":       RandomForestClassificationModel.load(str(MODELS_DIR / "RandomForest_mllib")),
}
print(f"Loaded {len(models)} models")

precision_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
recall_eval    = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

test_results = []
predictions_dict = {}

for name, model in models.items():
    preds = model.transform(test_df)
    predictions_dict[name] = preds

    accuracy  = acc_eval.evaluate(preds)
    f1        = f1_eval.evaluate(preds)
    precision = precision_eval.evaluate(preds)
    recall    = recall_eval.evaluate(preds)
    try:
        auc_val = auc_evaluator.evaluate(preds)
    except Exception:
        auc_val = None

    test_results.append({
        "model": name, "accuracy": round(accuracy, 4), "f1": round(f1, 4),
        "precision": round(precision, 4), "recall": round(recall, 4),
        "auc": round(auc_val, 4) if auc_val else None,
    })

results_df = pd.DataFrame(test_results)
print(results_df.to_string(index=False))

# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, preds) in zip(axes, predictions_dict.items()):
    pdf = preds.select("label", "prediction").toPandas()
    cm = confusion_matrix(pdf["label"], pdf["prediction"], labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Reliable", "Fake"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}\nAcc={results_df.loc[results_df['model']==name,'accuracy'].values[0]}")
plt.tight_layout()
plt.savefig(str(TABLEAU_DIR / "confusion_matrices.png"), dpi=150, bbox_inches="tight")
print("Saved confusion_matrices.png")

# ROC Curves
roc_data_all = []
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

for name, preds in predictions_dict.items():
    if "probability" in preds.columns:
        pdf = preds.select("label", F.element_at("probability", 2).alias("prob_1")).toPandas()
    else:
        from pyspark.ml.functions import vector_to_array
        pdf = preds.select("label", vector_to_array("rawPrediction").getItem(1).alias("prob_1")).toPandas()

    fpr, tpr, _ = roc_curve(pdf["label"], pdf["prob_1"])
    roc_auc_val = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_val:.3f})")
    for f, t in zip(fpr, tpr):
        roc_data_all.append({"model": name, "fpr": f, "tpr": t, "auc": roc_auc_val})

ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curves — Test Set"); ax_roc.legend()
plt.tight_layout()
plt.savefig(str(TABLEAU_DIR / "roc_curves.png"), dpi=150, bbox_inches="tight")
print("Saved roc_curves.png")

roc_df = pd.DataFrame(roc_data_all)
roc_df.to_csv(str(TABLEAU_DIR / "roc_data.csv"), index=False)
print(f"Exported {len(roc_df):,} ROC data points")

# Feature Importance (Random Forest)
rf_model = models["RandomForest"]
importances = rf_model.featureImportances.toArray()
top_k = 30
top_indices = np.argsort(importances)[::-1][:top_k]
top_scores  = importances[top_indices]

fi_df = pd.DataFrame({"feature_index": top_indices, "importance": top_scores, "rank": range(1, top_k+1)})
fi_df["feature_label"] = fi_df["feature_index"].apply(lambda x: f"hash_{x}")

fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
sns.barplot(data=fi_df, x="importance", y="feature_label", ax=ax_fi, palette="viridis")
ax_fi.set_title("Top 30 Features by Importance (Random Forest)")
ax_fi.set_xlabel("Gini Importance")
plt.tight_layout()
plt.savefig(str(TABLEAU_DIR / "feature_importance.png"), dpi=150, bbox_inches="tight")
print("Saved feature_importance.png")

fi_df.to_csv(str(TABLEAU_DIR / "feature_importance.csv"), index=False)

# Reverse hash lookup
feat_pipeline = PipelineModel.load(str(MODELS_DIR / "feature_pipeline"))
sample_df = spark.read.parquet(str(DATA_PARQUET / "news_articles")).limit(5000)
tokenized = feat_pipeline.stages[0].transform(sample_df)
filtered  = feat_pipeline.stages[1].transform(tokenized)

from pyspark.sql.functions import explode
words_df = filtered.select(explode("filtered_words").alias("word"))
unique_words = [row.word for row in words_df.distinct().collect()]

NUM_FEATURES = 2**14
hash_to_word = defaultdict(list)
for w in unique_words:
    h = hash(w) % NUM_FEATURES
    if h in set(top_indices):
        hash_to_word[h].append(w)

fi_df["words"] = fi_df["feature_index"].apply(lambda x: ", ".join(hash_to_word.get(x, ["unknown"])[:5]))
fi_df.to_csv(str(TABLEAU_DIR / "feature_importance_with_words.csv"), index=False)
print(fi_df[["rank", "importance", "words"]].head(15).to_string(index=False))

results_df.to_csv(str(TABLEAU_DIR / "test_metrics.csv"), index=False)
print("Exported test_metrics.csv")

test_df.unpersist()
print(f"\nNotebook 4 complete ({time.time()-t_nb4:.1f}s)")

# =====================================================================
# SCALABILITY EXPERIMENTS
# =====================================================================
print("\n" + "=" * 70)
print("  SCALABILITY EXPERIMENTS")
print("=" * 70)
t_scale = time.time()

scaling_results = []
data_fractions = [0.25, 0.5, 0.75, 1.0]

for frac in data_fractions:
    sample = spark.read.parquet(str(FEATURES / "train")).sample(fraction=frac, seed=42)
    sample.persist(StorageLevel.MEMORY_AND_DISK)
    n = sample.count()

    lr_test = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)
    t0 = time.time()
    lr_test.fit(sample)
    elapsed = time.time() - t0

    scaling_results.append({"data_fraction": frac, "num_rows": n, "train_time_s": round(elapsed, 2)})
    print(f"  frac={frac:.2f}  rows={n:>8,}  time={elapsed:.2f}s")
    sample.unpersist()

scaling_df = pd.DataFrame(scaling_results)
scaling_df.to_csv(str(TABLEAU_DIR / "scaling_experiments.csv"), index=False)
print(f"Exported scaling_experiments.csv")

# Scaling plot
fig_s, ax_s = plt.subplots(figsize=(8, 5))
ax_s.plot(scaling_df["num_rows"], scaling_df["train_time_s"], "o-", linewidth=2)
ax_s.set_xlabel("Number of Training Rows"); ax_s.set_ylabel("Training Time (s)")
ax_s.set_title("Scalability: LR Training Time vs Data Size")
plt.tight_layout()
plt.savefig(str(TABLEAU_DIR / "scaling_plot.png"), dpi=150, bbox_inches="tight")
print("Saved scaling_plot.png")
print(f"\nScalability experiments complete ({time.time()-t_scale:.1f}s)")

# =====================================================================
# TABLEAU EXPORT SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("  TABLEAU EXPORT SUMMARY")
print("=" * 70)
for f in sorted(TABLEAU_DIR.glob("*")):
    size = f.stat().st_size / 1024
    print(f"  {f.name:50s} {size:>8.1f} KB")

# =====================================================================
# DONE
# =====================================================================
total_time = time.time() - t_nb1
print(f"\n{'='*70}")
print(f"  ALL DONE — Total pipeline time: {total_time:.1f}s")
print(f"{'='*70}")

spark.stop()
