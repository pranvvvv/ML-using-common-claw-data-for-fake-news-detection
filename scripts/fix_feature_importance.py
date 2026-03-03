"""Pure-Python feature importance reverse hash lookup — no Spark workers needed."""
import os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import re, mmh3

# Basic English stopwords (avoids nltk dependency issues)
STOP_WORDS = set("i me my myself we our ours ourselves you your yours yourself yourselves "
    "he him his himself she her hers herself it its itself they them their theirs "
    "themselves what which who whom this that these those am is are was were be been "
    "being have has had having do does did doing a an the and but if or because as "
    "until while of at by for with about against between through during before after "
    "above below to from up down in out on off over under again further then once "
    "here there when where why how all both each few more most other some such no "
    "nor not only own same so than too very s t can will just don should now d ll m "
    "o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn "
    "shan shouldn wasn weren won wouldn".split())

ROOT        = Path(__file__).resolve().parent.parent
DATA_PARQUET= ROOT / "data" / "parquet"
MODELS_DIR  = ROOT / "data" / "models"
TABLEAU_DIR = ROOT / "tableau"
NUM_FEATURES= 2**14

# ── Load RF importances via Spark (only JVM, no Python workers) ──
os.environ["JAVA_HOME"]             = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot"
os.environ["HADOOP_HOME"]           = r"C:\hadoop"
os.environ["PATH"]                  = r"C:\hadoop\bin;" + os.environ.get("PATH","")
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel

spark = (SparkSession.builder.appName("FI_Fix").master("local[*]")
    .config("spark.driver.memory","2g")
    .config("spark.ui.showConsoleProgress","false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

rf = RandomForestClassificationModel.load(str(MODELS_DIR/"RandomForest_mllib"))
importances = rf.featureImportances.toArray()
spark.stop()
print("RF importances loaded.")

top_k = 30
top_idx = np.argsort(importances)[::-1][:top_k]
top_set = set(int(i) for i in top_idx)

# ── MurmurHash3 (matches Spark's HashingTF) ──
def spark_hash_bucket(word: str, num_features: int) -> int:
    """Spark 4.x HashingTF uses MurmurHash3_x86_32 with seed=42 on UTF-8 bytes."""
    h = mmh3.hash(word, seed=42, signed=True)
    return ((h % num_features) + num_features) % num_features

# ── Read articles with pyarrow (no Spark workers) ──
articles_path = DATA_PARQUET / "news_articles"
table = pq.read_table(str(articles_path), columns=["text"])
texts = table.column("text").to_pylist()
print(f"Read {len(texts)} articles.")

# ── Tokenize + stopword removal in pure Python ──
unique_words = set()
for text in texts:
    if text:
        tokens = re.findall(r'\b[a-z]{2,}\b', text.lower())
        for w in tokens:
            if w not in STOP_WORDS and len(w) > 1:
                unique_words.add(w)
print(f"Unique words: {len(unique_words)}")

# ── Map ALL words to hash buckets (debug: show bucket distribution) ──
all_buckets = defaultdict(list)
hash_to_words = defaultdict(list)
for w in unique_words:
    bucket = spark_hash_bucket(w, NUM_FEATURES)
    all_buckets[bucket].append(w)
    if bucket in top_set:
        hash_to_words[bucket].append(w)

print(f"Words mapped to {len(all_buckets)} unique buckets out of {NUM_FEATURES}")
print(f"Top-30 buckets with word hits: {len(hash_to_words)}")

# Sort by importance within each bucket
for k in hash_to_words:
    hash_to_words[k].sort()

# ── Build final table ──
fi_df = pd.DataFrame({
    "feature_index": top_idx,
    "importance": importances[top_idx],
    "rank": range(1, top_k + 1)
})
fi_df["words"] = fi_df["feature_index"].apply(
    lambda x: ", ".join(hash_to_words.get(x, ["--"])[:5])
)
fi_df["feature_label"] = fi_df.apply(
    lambda r: (r["words"][:35] if r["words"] != "--" else f"hash_{r['feature_index']}"), axis=1
)

print("\nTop 30 Features (Random Forest):")
print(fi_df[["rank", "importance", "words"]].to_string(index=False))

# ── Save CSV ──
fi_df.to_csv(str(TABLEAU_DIR/"feature_importance.csv"), index=False)
fi_df.to_csv(str(TABLEAU_DIR/"feature_importance_with_words.csv"), index=False)

# ── Plot ──
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=fi_df, x="importance", y="feature_label", ax=ax, palette="viridis")
ax.set_title("Top 30 Features by Importance (Random Forest)", fontsize=14)
ax.set_xlabel("Gini Importance"); ax.set_ylabel("")
plt.tight_layout()
plt.savefig(str(TABLEAU_DIR/"feature_importance.png"), dpi=150, bbox_inches="tight")
print("\nSaved feature_importance.png + CSVs")
