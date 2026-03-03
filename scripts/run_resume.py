"""Resume pipeline from NB3 — data + features already saved."""
import os, sys, time, pickle, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
warnings.filterwarnings("ignore")

os.environ["JAVA_HOME"]             = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot"
os.environ["HADOOP_HOME"]           = r"C:\hadoop"
os.environ["PATH"]                  = r"C:\hadoop\bin;" + os.environ.get("PATH","")
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import (
    LogisticRegression, LinearSVC, RandomForestClassifier,
    LogisticRegressionModel, LinearSVCModel, RandomForestClassificationModel,
)
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark import StorageLevel

ROOT        = Path(__file__).resolve().parent.parent
DATA_PARQUET= ROOT / "data" / "parquet"
FEATURES    = DATA_PARQUET / "features"
MODELS_DIR  = ROOT / "data" / "models"
TABLEAU_DIR = ROOT / "tableau"
for d in [FEATURES, MODELS_DIR, TABLEAU_DIR]: d.mkdir(parents=True, exist_ok=True)

spark = (SparkSession.builder.appName("FakeNewsDetection").master("local[*]")
    .config("spark.driver.memory","4g").config("spark.sql.shuffle.partitions","8")
    .config("spark.sql.execution.arrow.pyspark.enabled","true")
    .config("spark.sql.adaptive.enabled","true")
    .config("spark.ui.showConsoleProgress","false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
print(f"Spark {spark.version}")

# ── NB3 — MODEL TRAINING ──
print("\n" + "="*70 + "\n  NOTEBOOK 3 — MODEL TRAINING\n" + "="*70)
t_nb3 = time.time()

train_df = spark.read.parquet(str(FEATURES/"train")).persist(StorageLevel.MEMORY_AND_DISK)
val_df   = spark.read.parquet(str(FEATURES/"val")).persist(StorageLevel.MEMORY_AND_DISK)
test_df  = spark.read.parquet(str(FEATURES/"test")).persist(StorageLevel.MEMORY_AND_DISK)
print(f"Train: {train_df.count():>8,} | Val: {val_df.count():>6,} | Test: {test_df.count():>6,}")

auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_eval  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

cv_results = {}

# LR with CV
lr = LogisticRegression(featuresCol="features",labelCol="label",maxIter=100,family="binomial")
lr_grid = (ParamGridBuilder().addGrid(lr.regParam,[0.01,0.1]).addGrid(lr.elasticNetParam,[0.0,0.5]).build())
cv_lr = CrossValidator(estimator=lr, estimatorParamMaps=lr_grid, evaluator=auc_eval, numFolds=3, parallelism=2, seed=42)
print("\nTraining LR (4 combos x 3 folds)...")
t0=time.time(); cv_model_lr=cv_lr.fit(train_df); t_lr=time.time()-t0
cv_results["LogisticRegression"] = {"best_model":cv_model_lr.bestModel, "best_auc":max(cv_model_lr.avgMetrics),
    "train_time":t_lr, "avg_metrics":cv_model_lr.avgMetrics}
print(f"  LR done: AUC={max(cv_model_lr.avgMetrics):.4f} in {t_lr:.0f}s")

# SVC with CV
svc = LinearSVC(featuresCol="features",labelCol="label",maxIter=100)
svc_grid = (ParamGridBuilder().addGrid(svc.regParam,[0.01,0.1]).build())
cv_svc = CrossValidator(estimator=svc, estimatorParamMaps=svc_grid, evaluator=auc_eval, numFolds=3, parallelism=2, seed=42)
print("\nTraining SVC (2 combos x 3 folds)...")
t0=time.time(); cv_model_svc=cv_svc.fit(train_df); t_svc=time.time()-t0
cv_results["LinearSVC"] = {"best_model":cv_model_svc.bestModel, "best_auc":max(cv_model_svc.avgMetrics),
    "train_time":t_svc, "avg_metrics":cv_model_svc.avgMetrics}
print(f"  SVC done: AUC={max(cv_model_svc.avgMetrics):.4f} in {t_svc:.0f}s")

# RF direct fit
rf = RandomForestClassifier(featuresCol="features",labelCol="label",numTrees=100,maxDepth=5,maxBins=32,seed=42)
print("\nTraining RF (direct fit, 100 trees, depth=5)...")
t0=time.time(); rf_model=rf.fit(train_df); t_rf=time.time()-t0
rf_auc = auc_eval.evaluate(rf_model.transform(train_df))
cv_results["RandomForest"] = {"best_model":rf_model, "best_auc":rf_auc,
    "train_time":t_rf, "avg_metrics":[rf_auc]}
print(f"  RF done: AUC={rf_auc:.4f} in {t_rf:.0f}s")

# Validation metrics
print("\n--- Validation Results ---")
val_metrics = []
for name, res in cv_results.items():
    preds = res["best_model"].transform(val_df)
    acc = acc_eval.evaluate(preds); f1 = f1_eval.evaluate(preds)
    try: auc_v = auc_eval.evaluate(preds)
    except: auc_v = None
    val_metrics.append({"model":name,"accuracy":round(acc,4),"f1":round(f1,4),
        "auc":round(auc_v,4) if auc_v else "N/A","train_time_s":round(res["train_time"],1)})
    print(f"  {name:25s} Acc={acc:.4f} F1={f1:.4f} AUC={auc_v}")
val_metrics_df = pd.DataFrame(val_metrics)

# Save models
for name, res in cv_results.items():
    res["best_model"].write().overwrite().save(str(MODELS_DIR/f"{name}_mllib"))
    meta = {"model_name":name,"best_auc":res["best_auc"],"train_time":res["train_time"],"avg_metrics":res["avg_metrics"]}
    with open(MODELS_DIR/f"{name}_cvmodel.pkl","wb") as fh: pickle.dump(meta,fh)
val_metrics_df.to_csv(str(TABLEAU_DIR/"model_comparison.csv"),index=False)
print(f"\nNB3 done ({time.time()-t_nb3:.0f}s)")

train_df.unpersist(); val_df.unpersist(); test_df.unpersist()

# ── NB4 — EVALUATION ──
print("\n" + "="*70 + "\n  NOTEBOOK 4 — EVALUATION\n" + "="*70)
t_nb4 = time.time()

test_df = spark.read.parquet(str(FEATURES/"test")).persist(StorageLevel.MEMORY_AND_DISK)
print(f"Test: {test_df.count():,}")

models = {
    "LogisticRegression": LogisticRegressionModel.load(str(MODELS_DIR/"LogisticRegression_mllib")),
    "LinearSVC":          LinearSVCModel.load(str(MODELS_DIR/"LinearSVC_mllib")),
    "RandomForest":       RandomForestClassificationModel.load(str(MODELS_DIR/"RandomForest_mllib")),
}

prec_eval = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction",metricName="weightedPrecision")
rec_eval  = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction",metricName="weightedRecall")

test_results = []; predictions_dict = {}
for name, model in models.items():
    preds = model.transform(test_df); predictions_dict[name] = preds
    acc=acc_eval.evaluate(preds); f1=f1_eval.evaluate(preds)
    prec=prec_eval.evaluate(preds); rec=rec_eval.evaluate(preds)
    try: auc_v=auc_eval.evaluate(preds)
    except: auc_v=None
    test_results.append({"model":name,"accuracy":round(acc,4),"f1":round(f1,4),
        "precision":round(prec,4),"recall":round(rec,4),"auc":round(auc_v,4) if auc_v else None})

results_df = pd.DataFrame(test_results)
print(results_df.to_string(index=False))

# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, preds) in zip(axes, predictions_dict.items()):
    pdf = preds.select("label", "prediction").toPandas()
    cm = confusion_matrix(pdf["label"], pdf["prediction"], labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Reliable","Fake"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}\nAcc={results_df.loc[results_df['model']==name,'accuracy'].values[0]}")
plt.tight_layout()
plt.savefig(str(TABLEAU_DIR/"confusion_matrices.png"), dpi=150, bbox_inches="tight")
print("Saved confusion_matrices.png")

# ROC Curves
from pyspark.ml.functions import vector_to_array
roc_data_all = []; fig_roc, ax_roc = plt.subplots(figsize=(8,6))
for name, preds in predictions_dict.items():
    if "probability" in preds.columns:
        pdf = preds.select("label", vector_to_array("probability").getItem(1).alias("prob_1")).toPandas()
    else:
        pdf = preds.select("label", vector_to_array("rawPrediction").getItem(1).alias("prob_1")).toPandas()
    fpr, tpr, _ = roc_curve(pdf["label"], pdf["prob_1"])
    roc_auc_val = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_val:.3f})")
    for f, t in zip(fpr, tpr):
        roc_data_all.append({"model":name,"fpr":f,"tpr":t,"auc":roc_auc_val})
ax_roc.plot([0,1],[0,1],"k--",alpha=0.3,label="Random")
ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curves — Test Set"); ax_roc.legend()
plt.tight_layout()
plt.savefig(str(TABLEAU_DIR/"roc_curves.png"), dpi=150, bbox_inches="tight")
pd.DataFrame(roc_data_all).to_csv(str(TABLEAU_DIR/"roc_data.csv"),index=False)
print("Saved roc_curves.png + roc_data.csv")

# Feature Importance
rf_model = models["RandomForest"]
importances = rf_model.featureImportances.toArray()
top_k = 30; top_idx = np.argsort(importances)[::-1][:top_k]
fi_df = pd.DataFrame({"feature_index":top_idx,"importance":importances[top_idx],"rank":range(1,top_k+1)})
fi_df["feature_label"] = fi_df["feature_index"].apply(lambda x: f"hash_{x}")

fig_fi, ax_fi = plt.subplots(figsize=(10,8))
sns.barplot(data=fi_df, x="importance", y="feature_label", ax=ax_fi, palette="viridis")
ax_fi.set_title("Top 30 Features by Importance (Random Forest)"); ax_fi.set_xlabel("Gini Importance")
plt.tight_layout()
plt.savefig(str(TABLEAU_DIR/"feature_importance.png"), dpi=150, bbox_inches="tight")
fi_df.to_csv(str(TABLEAU_DIR/"feature_importance.csv"),index=False)
print("Saved feature_importance.png + .csv")

# Reverse hash lookup
feat_pipeline = PipelineModel.load(str(MODELS_DIR/"feature_pipeline"))
sample_df = spark.read.parquet(str(DATA_PARQUET/"news_articles")).limit(5000)
tokenized = feat_pipeline.stages[0].transform(sample_df)
filtered  = feat_pipeline.stages[1].transform(tokenized)
from pyspark.sql.functions import explode
unique_words = [r.word for r in filtered.select(explode("filtered_words").alias("word")).distinct().collect()]
NUM_FEATURES = 2**14
hash_to_word = defaultdict(list)
for w in unique_words:
    h = hash(w) % NUM_FEATURES
    if h in set(top_idx): hash_to_word[h].append(w)
fi_df["words"] = fi_df["feature_index"].apply(lambda x: ", ".join(hash_to_word.get(x,["unknown"])[:5]))
fi_df.to_csv(str(TABLEAU_DIR/"feature_importance_with_words.csv"),index=False)
print(fi_df[["rank","importance","words"]].head(15).to_string(index=False))

results_df.to_csv(str(TABLEAU_DIR/"test_metrics.csv"),index=False)
test_df.unpersist()
print(f"\nNB4 done ({time.time()-t_nb4:.0f}s)")

# ── SCALABILITY EXPERIMENTS ──
print("\n" + "="*70 + "\n  SCALABILITY EXPERIMENTS\n" + "="*70)
t_s = time.time()
scaling_results = []
for frac in [0.25, 0.5, 0.75, 1.0]:
    sample = spark.read.parquet(str(FEATURES/"train")).sample(fraction=frac, seed=42)
    sample.persist(StorageLevel.MEMORY_AND_DISK); n=sample.count()
    lr_t = LogisticRegression(featuresCol="features",labelCol="label",maxIter=20)
    t0=time.time(); lr_t.fit(sample); el=time.time()-t0
    scaling_results.append({"data_fraction":frac,"num_rows":n,"train_time_s":round(el,2)})
    print(f"  frac={frac:.2f} rows={n:>8,} time={el:.2f}s"); sample.unpersist()
scaling_df = pd.DataFrame(scaling_results)
scaling_df.to_csv(str(TABLEAU_DIR/"scaling_experiments.csv"),index=False)
fig_s, ax_s = plt.subplots(figsize=(8,5))
ax_s.plot(scaling_df["num_rows"],scaling_df["train_time_s"],"o-",linewidth=2)
ax_s.set_xlabel("Number of Training Rows"); ax_s.set_ylabel("Training Time (s)")
ax_s.set_title("Scalability: LR Training Time vs Data Size")
plt.tight_layout(); plt.savefig(str(TABLEAU_DIR/"scaling_plot.png"),dpi=150,bbox_inches="tight")
print(f"\nScalability done ({time.time()-t_s:.0f}s)")

# ── SUMMARY ──
print("\n" + "="*70 + "\n  TABLEAU EXPORT SUMMARY\n" + "="*70)
for f in sorted(TABLEAU_DIR.glob("*")):
    print(f"  {f.name:50s} {f.stat().st_size/1024:>8.1f} KB")
total = time.time() - t_nb3
print(f"\n{'='*70}\n  ALL DONE — Total: {total:.0f}s\n{'='*70}")
spark.stop()
