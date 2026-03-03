"""Resume from NB4 — models already saved by run_resume.py."""
import os, sys, time, warnings
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
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import (
    LogisticRegressionModel, LinearSVCModel, RandomForestClassificationModel,
)
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark import StorageLevel

ROOT        = Path(__file__).resolve().parent.parent
DATA_PARQUET= ROOT / "data" / "parquet"
FEATURES    = DATA_PARQUET / "features"
MODELS_DIR  = ROOT / "data" / "models"
TABLEAU_DIR = ROOT / "tableau"

spark = (SparkSession.builder.appName("FakeNewsDetection").master("local[*]")
    .config("spark.driver.memory","4g").config("spark.sql.shuffle.partitions","8")
    .config("spark.sql.execution.arrow.pyspark.enabled","true")
    .config("spark.sql.adaptive.enabled","true")
    .config("spark.ui.showConsoleProgress","false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
print(f"Spark {spark.version}")

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

auc_eval  = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
acc_eval  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_eval   = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
prec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
rec_eval  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

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

# ROC Curves (PySpark 4.x: vector_to_array needed)
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
from pyspark.ml.classification import LogisticRegression
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
print(f"\nALL DONE ({time.time()-t_nb4:.0f}s total)")
spark.stop()
