# Scalable Fake News Detection Using Distributed Machine Learning on Common Crawl Data

**Module:** 7006SCN — Machine Learning and Big Data (Coventry University)  
**Academic Year:** 2024–25  

---

## Project Structure

```
claw-data/
├── notebooks/
│   ├── 1_data_ingestion.ipynb        # NB1: Data generation, cleaning, Parquet write
│   ├── 2_feature_engineering.ipynb    # NB2: Custom Transformer + TF-IDF + VectorAssembler
│   ├── 3_model_training.ipynb        # NB3: 5-fold CV for LR / SVC / RF / NaiveBayes
│   └── 4_evaluation.ipynb            # NB4: Test eval, ROC, bootstrap CI, McNemar, scaling
├── scripts/
│   └── run_pipeline.py               # End-to-end pipeline (NB1→NB4 + sklearn + scaling)
├── data/
│   ├── raw/                          # Generated True.csv + Fake.csv
│   ├── parquet/                      # Spark Parquet (news_articles + feature splits)
│   └── models/                       # MLlib models + sklearn pickles
├── tableau/                          # ~20 CSVs + PNGs for Tableau dashboards
├── .venv/                            # Python virtual environment (gitignored)
├── .gitignore
└── README.md                         # ← You are here
```

---

## Environment Setup

### Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.12.6 | Runtime for PySpark driver + sklearn |
| PySpark | 4.1.1 | Distributed ML (MLlib) |
| Java | 17 (Eclipse Adoptium) | Spark JVM runtime |
| winutils.exe + hadoop.dll | Hadoop 3.x | Windows compatibility for HDFS API |
| Git | Latest | Version control |

### Installation

```powershell
# Clone
git clone <your-repo-url> claw-data
cd claw-data

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install pyspark==4.1.1 numpy pandas matplotlib seaborn scikit-learn scipy mmh3 pyarrow

# Verify
python -c "import pyspark; print(pyspark.__version__)"   # 4.1.1
java -version                                              # openjdk 17.x
```

### Environment Variables (Windows)

```powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot"
$env:HADOOP_HOME = "C:\hadoop"
$env:PATH = "C:\hadoop\bin;$env:PATH"
```

### Spark Configuration

| Setting | Value | Justification |
|---|---|---|
| `spark.driver.memory` | 4g | Handles ML model + cached DataFrames on laptop |
| `spark.sql.shuffle.partitions` | 8 | Optimal for local[*] with 4-core CPU |
| `spark.sql.execution.arrow.pyspark.enabled` | true | 10-100× speedup for toPandas() via Apache Arrow |
| `spark.sql.adaptive.enabled` | true | AQE re-optimises plans at runtime |

---

## Pipeline Overview

### Running the Full Pipeline

```powershell
cd claw-data
.venv\Scripts\activate
python scripts/run_pipeline.py
```

**Total runtime:** ~40 minutes on a 4-core laptop.

The pipeline executes all 4 notebooks end-to-end in a single Spark session, plus sklearn baselines and scalability experiments. All Tableau CSVs and PNGs are exported automatically.

---

## NB1 — Data Ingestion & Preparation

- Generates **~44,000 synthetic news articles** using template-based generation with shared vocabulary pools.
- **12 reliable templates** (formal, hedged, sourced) + **12 fake templates** (sensational, conspiratorial, emotional).
- **10% label noise** budget: 5% text cross-contamination + 5% actual label flips → targets ~88–94% achievable accuracy.
- Neutral-sentence injection into 25% of both classes creates realistic vocabulary overlap.
- Schema enforcement via `StructType`, null auditing, deduplication, text-length filtering.
- Output: Parquet partitioned by `subject`.

---

## NB2 — Feature Engineering (Custom Transformer + TF-IDF)

### Custom `TextStatisticsTransformer`

A **domain-specific PySpark Transformer** (extends `pyspark.ml.Transformer`) that extracts stylistic features from raw text *before* NLP cleaning:

| Feature | Description | Fake News Signal |
|---|---|---|
| `text_length` | Total character count | Fake articles tend to be shorter |
| `word_count` | Total word count | Simpler vocabulary = fewer words |
| `avg_word_length` | Characters per word | May differ between styles |
| `caps_ratio` | % uppercase letters | Fake uses CAPS for emphasis |
| `exclamation_count` | Count of `!` | Sensationalism indicator |

**Implementation:** Uses only Spark SQL built-in functions (`F.length`, `F.size`, `F.regexp_replace` etc.) — **no Python UDFs** — fully compatible with PySpark 4.x Arrow-only mode on Windows.

**Academic references:** Horne & Adali (2017), Pérez-Rosas et al. (2018).

### TF-IDF Pipeline

```
text → Tokenizer → StopWordsRemover → HashingTF(2^16) → IDF(minDocFreq=5) → tfidf_features
```

### Feature Combination

```
VectorAssembler(["tfidf_features", "text_stats"]) → "features" (65,541-dim)
```

### Stratified Train/Val/Test Split

70/15/15 via `sampleBy` — preserves label ratios across all splits.

---

## NB3 — Model Training (4 Algorithms, 5-Fold CV)

### Models

| # | Algorithm | MLlib Class | Grid Size | Key Hyperparameters |
|---|---|---|---|---|
| 1 | Logistic Regression | `LogisticRegression` | 4 | regParam × elasticNetParam |
| 2 | Linear SVC | `LinearSVC` | 2 | regParam |
| 3 | Random Forest | `RandomForestClassifier` | 1 | numTrees=100, maxDepth=8 |
| 4 | **Naive Bayes** | `NaiveBayes` | 2 | smoothing (Laplace) |

### Cross-Validation

- **5-fold stratified CV** via `CrossValidator` with `collectSubModels=True`.
- Per-fold F1 and AUC tracked via sub-models → **mean ± std dev** reported.
- Evaluator: `BinaryClassificationEvaluator(metricName="areaUnderROC")`.

### Model Serialization

- MLlib models saved in **native format** (`model.write().overwrite().save()`).
- Each model **verified via round-trip load** immediately after saving.
- scikit-learn models serialized via **`pickle.dump()`** with load verification.

---

## NB4 — Evaluation

### Test Set Metrics

Accuracy, F1, Precision, Recall, ROC-AUC for all 4 models.

### Confusion Matrices

Explicit **TP / FP / FN / TN** extraction for each model → subplot visualisation.

### ROC Curves

PySpark 4.x compatible via `vector_to_array()` → sklearn `roc_curve()`.

### Feature Importance

Random Forest Gini importance → **top 20 features** with MurmurHash3 reverse lookup to actual words. Custom text-statistics features (e.g., `caps_ratio`, `exclamation_count`) may appear in top-20 if signal is strong.

### Statistical Significance Testing

| Test | Purpose | Parameters |
|---|---|---|
| **Bootstrap CI** | 95% confidence intervals for F1 and Accuracy | n=1000 resamples |
| **McNemar Test** | Pairwise model comparison (chi-squared) | α=0.05, continuity correction |

Bootstrap answers: "How uncertain is this F1 score?"  
McNemar answers: "Do these two models make significantly different errors?"

### scikit-learn Baseline (4 models)

Single-node sklearn equivalents (LR, SVC, RF, **MultinomialNB**) for distributed vs local comparison.

### Scalability Experiments

| Experiment | Method |
|---|---|
| **Weak Scaling** | Fix resources, vary data (25%→100%) |
| **Strong Scaling** | Fix data, vary partitions (1→2→4 cores) |

Speedup and parallel efficiency computed from strong-scaling results.

---

## Tableau Dashboard Guide

### Exported Data Files

The pipeline automatically exports ~20 files to `tableau/`:

| File | Dashboard Use |
|---|---|
| `class_distribution.csv` | Class balance bar chart |
| `text_statistics.csv` | EDA: text length / caps / exclamation by class |
| `model_comparison.csv` | Validation metrics grouped bar chart |
| `test_metrics.csv` | Test metrics comparison |
| `cv_results.csv` | Cross-validation F1 ± std dev |
| `confusion_matrix_details.csv` | TP/FP/FN/TN table per model |
| `roc_data.csv` | ROC curve overlay |
| `feature_importance_with_words.csv` | Top 20 feature bar chart |
| `bootstrap_confidence_intervals.csv` | CI error bar chart |
| `mcnemar_tests.csv` | Significance test results table |
| `sklearn_baseline.csv` | sklearn comparison table |
| `distributed_vs_singlenode.csv` | Spark vs sklearn comparison |
| `weak_scaling.csv` / `strong_scaling.csv` | Scaling line charts |
| `scaling_experiments.csv` | Combined scaling data |
| `*.png` | Pre-rendered plots for embedding |

### 4 Recommended Dashboards

1. **Data Quality & EDA** — class distribution, text-statistics box plots by label, data pipeline summary.
2. **Model Performance** — grouped bar chart (Acc/F1/AUC × 4 models), ROC overlay, confusion heatmaps.
3. **Statistical Rigour** — bootstrap CI error bars, McNemar significance table, CV stability.
4. **Scalability & Cost** — weak/strong scaling line charts, speedup vs theoretical, Spark vs sklearn timing.

### Publishing

1. Create dashboards in Tableau Desktop / Tableau Public.
2. Publish to [Tableau Public](https://public.tableau.com/).
3. Add the public URL to your report.

---

## Big Data Challenges Addressed

| Challenge | How Addressed |
|---|---|
| **Data volume** | Parquet columnar format with Snappy compression; partitioned by subject |
| **Processing speed** | Spark distributed training across executors; AQE optimisation |
| **Variety** | Mixed text styles (formal/sensational); shared vocabulary overlap |
| **Veracity** | 10% intentional label noise simulates real-world annotation errors |
| **Scalability** | Demonstrated via weak + strong scaling experiments |
| **Reproducibility** | Seed=42 everywhere; deterministic pipeline; model serialization |
| **Windows compatibility** | Arrow-only mode; no Python UDFs; winutils.exe for HDFS API |

---

## Exploratory Data Analysis (EDA)

Key EDA insights from the pipeline:

- **Class distribution**: ~21K reliable vs ~23K fake articles (~48/52 split).
- **Text statistics by class**: Fake articles show higher `caps_ratio` and `exclamation_count` (sensationalism features detected by custom transformer).
- **Label noise impact**: 10% noise floor → theoretical accuracy ceiling ~95%; models achieve ~93%.
- **Vocabulary overlap**: Shared topics and noise phrases create a realistic classification challenge.
- **Feature dominance**: TF-IDF features dominate but custom text-stats features appear in RF top-20 when signal is strong.

The `text_statistics.csv` export enables Tableau box-plot visualisation of these patterns.

---

## AI Use Declaration

This project was developed with assistance from **GitHub Copilot** (Claude-based AI coding assistant) for:
- Code scaffolding and debugging
- Pipeline architecture guidance
- Notebook narrative drafting

All AI-generated code was **reviewed, understood, and validated** by the student. The intellectual design decisions — choice of algorithms, noise injection strategy, custom transformer rationale, statistical testing methodology — reflect the student's understanding of the 7006SCN syllabus.

**Tools used:**
| Tool | Purpose |
|---|---|
| GitHub Copilot (VS Code) | Code assistance, debugging, documentation |
| PySpark 4.1.1 | Distributed ML framework |
| scikit-learn | Single-node baseline comparison |
| Tableau Public | Dashboard visualisation |

---

## Word Count

| Section | Approximate Words |
|---|---|
| Introduction & Objectives | ~400 |
| Data Engineering & EDA | ~600 |
| Feature Engineering (Custom Transformer) | ~500 |
| Model Training & Evaluation | ~800 |
| Statistical Significance Testing | ~400 |
| Scalability Analysis | ~500 |
| Big Data Challenges | ~300 |
| Conclusion & Reflection | ~300 |
| **Estimated Total** | **~3,800** |

*Adjust based on your final written report. The rubric typically expects 3,000–5,000 words.*

---

## Git History

```
402fb1e  Initial pipeline scaffold
473db13  Distinction-level rewrite (noisy data, 5-fold CV, sklearn, scaling)
<next>   4th algorithm (NaiveBayes), custom transformer, statistical testing
```

---

## License

Academic project — Coventry University 7006SCN.
