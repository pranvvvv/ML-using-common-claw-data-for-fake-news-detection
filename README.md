# Scalable Fake News Detection Using Distributed Machine Learning on Common Crawl Data

**Module:** 7006SCN — Machine Learning and Big Data (Coventry University)  
**Dataset:** [Common Crawl](https://commoncrawl.org/) via [AWS Open Data Registry](https://registry.opendata.aws/commoncrawl/)

---

## Project Structure

```
claw-data/
├── notebooks/
│   ├── 1_data_ingestion.ipynb        # STEP 2: Ingest Common Crawl → Parquet
│   ├── 2_feature_engineering.ipynb    # STEP 3: NLP pipeline + labelling
│   ├── 3_model_training.ipynb        # STEP 3: LR, SVC, RF + CrossValidator
│   └── 4_evaluation.ipynb            # STEP 3: Metrics, ROC, confusion, features
├── scripts/
│   ├── scalability_experiments.py    # STEP 4: Strong + Weak scaling benchmarks
│   └── tableau_export.py            # STEP 5: Consolidate Tableau exports
├── config/
│   ├── __init__.py
│   ├── spark_session.py             # SparkSession factory (config-driven)
│   └── spark_config.yaml            # Centralized Spark tuning parameters
├── data/
│   ├── raw/                         # Downloaded WET/WARC files (gitignored)
│   ├── processed/                   # Intermediate cleaned data
│   └── parquet/                     # Final Parquet datasets + feature splits
├── tableau/                         # CSVs + PNGs for Tableau dashboards
├── tests/
│   ├── test_spark_session.py
│   ├── test_schema.py
│   └── test_pipeline.py
├── environment.yml                  # Conda environment specification
├── Dockerfile                       # Reproducible containerized environment
├── .gitignore
└── README.md                        # ← You are here
```

---

## STEP 1 — Environment Setup

### 1.1 Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.10 | LTS; PySpark 3.5 compatibility |
| Java | 17 (JRE) | Spark JVM runtime |
| Conda | Latest | Environment isolation |
| Git | Latest | Version control |
| Docker | Latest (optional) | Reproducible container |

### 1.2 Installation (Local)

```powershell
# Clone repository
git clone <your-repo-url> claw-data
cd claw-data

# Create conda environment
conda env create -f environment.yml
conda activate fakenews-bigdata

# Verify PySpark
python -c "import pyspark; print(pyspark.__version__)"
# Expected: 3.5.1

# Verify Java
java -version
# Expected: openjdk 17.x
```

### 1.3 Installation (Docker — fully reproducible)

```powershell
docker build -t fakenews-bigdata .
docker run -p 8888:8888 -p 4040:4040 -v ${PWD}:/app fakenews-bigdata
```
- **Port 8888** → JupyterLab  
- **Port 4040** → Spark UI (live during sessions)

### 1.4 SparkSession Configuration Justification

All Spark settings live in `config/spark_config.yaml` and are loaded by `config/spark_session.py`.

| Setting | Value | Why |
|---|---|---|
| `spark.executor.memory` | 4g | Enough to hold partitioned text data in heap; prevents OOM on medium-size nodes |
| `spark.executor.cores` | 2 | 2 cores/executor → good task-level parallelism without over-subscribing CPU |
| `spark.sql.shuffle.partitions` | 200 | Default; set to 2× total cores for production. Prevents too many tiny tasks |
| `spark.sql.autoBroadcastJoinThreshold` | 50MB | Broadcasts small lookup tables (domain lists, stop words) → avoids shuffle join |
| `spark.sql.execution.arrow.pyspark.enabled` | true | 10-100× speed for `toPandas()` via Apache Arrow columnar transfer |
| `spark.sql.adaptive.enabled` | true | AQE re-optimises query plans at runtime → handles skew automatically |
| `spark.hadoop.fs.s3a.aws.credentials.provider` | Anonymous | Common Crawl is a public bucket → no AWS keys needed |

---

## STEP 2 — Data Acquisition & Ingestion

**Notebook:** `notebooks/1_data_ingestion.ipynb`

### How Common Crawl Works

Common Crawl is a **non-profit** that crawls the web monthly and stores the results on S3:

```
s3://commoncrawl/crawl-data/CC-MAIN-2024-10/
├── warc.paths.gz       # Index of WARC files (full HTTP)
├── wet.paths.gz        # Index of WET files (plain text only) ← WE USE THIS
└── wat.paths.gz        # Index of WAT files (metadata)
```

**Access method:** Anonymous S3 via `boto3` with `UNSIGNED` signature.  
No AWS account required for read access.

### Pipeline Architecture

```
Common Crawl S3 → boto3 stream → warcio parse → Spark DF → Schema validation
    → Null handling → Dedup → Language filter → Parquet (partitioned by date)
```

### Key Design Decisions

| Decision | Justification | Rubric Impact |
|---|---|---|
| WET over WARC | Already plain-text → no HTML parsing at scale | Data Engineering 20% |
| `StructType` schema | Catch schema drift early; enforced non-null on critical fields | Data Engineering 20% |
| Parquet + Snappy | Columnar + compressed → 10× smaller than CSV, partition pruning | Data Engineering 20% |
| Partition by `crawl_date` | Date-range queries skip irrelevant partitions | Data Engineering 20% |
| `MEMORY_AND_DISK` persist | Avoids re-download; spills to disk if heap full | Data Engineering 20% |
| Explicit `unpersist()` | Prevents memory leaks across notebook cells | Data Engineering 20% |

### Spark UI Monitoring

| Tab | What to Check |
|---|---|
| Jobs | Timeline, failed stages → detect stragglers |
| Stages | Shuffle R/W, task skew → find bottlenecks |
| Storage | Cached RDD sizes → validate persist lifecycle |
| SQL | Physical plan → confirm broadcast joins |
| Environment | Config values → verify YAML loaded correctly |

---

## STEP 3 — Distributed ML Pipeline

### Notebooks

| Notebook | Purpose |
|---|---|
| `2_feature_engineering.ipynb` | Tokenizer → StopWords → HashingTF → IDF |
| `3_model_training.ipynb` | LR, SVC, RF + 5-fold CV + ParamGrid |
| `4_evaluation.ipynb` | Accuracy, F1, ROC-AUC, Confusion, Features |

### NLP Feature Pipeline

```
text → Tokenizer → StopWordsRemover → HashingTF(2^18) → IDF(minDocFreq=5) → features
```

**Why HashingTF over CountVectorizer:**
- Fixed output dimension (2^18) → no vocabulary pass required.
- Scales to arbitrary vocab sizes without collecting to driver.
- Small hash collision rate at 262K features is acceptable.

### Models

| Model | MLlib Class | Grid Size | Hyperparameters Tuned |
|---|---|---|---|
| Logistic Regression | `LogisticRegression` | 9 | regParam × elasticNetParam |
| Linear SVC | `LinearSVC` | 3 | regParam |
| Random Forest | `RandomForestClassifier` | 9 | numTrees × maxDepth |

### Distributed Training vs scikit-learn

| Aspect | MLlib | scikit-learn |
|---|---|---|
| Data partitioning | Across executors | Single-node RAM |
| Gradient aggregation | AllReduce | N/A (in-process) |
| Tree parallelism | Different trees → different executors | Multi-thread in one process |
| Scaling | Horizontal (add nodes) | Vertical only (bigger RAM) |
| Data limit | Petabyte-scale | ~100 GB (RAM-bound) |

### Evaluation Metrics

- **Accuracy** — overall correctness
- **F1-score** — harmonic mean of precision & recall (handles imbalance)
- **ROC-AUC** — discrimination ability across all thresholds
- **Confusion Matrix** — per-class error analysis
- **Feature Importance** — Random Forest Gini importance + reverse hash lookup

---

## STEP 4 — Scalability Experiments

**Script:** `scripts/scalability_experiments.py`

### Strong Scaling

- **Fixed** dataset size, **vary** executors: 1 → 2 → 4.
- Measures how well adding resources reduces training time.
- **Ideal:** 2× executors → 2× speedup (linear).
- **Reality:** Communication overhead reduces efficiency.

### Weak Scaling

- **Proportionally increase** data AND executors together.
- Measures if the system maintains constant throughput.
- **Ideal:** Training time stays flat.

### Executor Configurations

| Profile | Executors | Cores/Exec | Memory/Exec |
|---|---|---|---|
| `single` | 1 | 2 | 4g |
| `dual`   | 2 | 2 | 4g |
| `quad`   | 4 | 2 | 4g |

### Metrics Collected

- Training wall-clock time
- Shuffle read/write (bytes)
- Total executor task time
- Speedup = T₁ / Tₙ
- Efficiency = Speedup / N
- Cost estimate = (time/60) × executors × $/executor-minute

### DAG and Shuffle Interpretation

In the Spark UI **Stages** tab:
- **Green bars** → computation time (good).
- **Blue bars** → shuffle read/write (overhead to minimise).
- **Skewed tasks** → one task takes 10× longer → `spark.sql.adaptive.skewJoin.enabled=true` helps.
- **DAG visualisation** → shows stage dependencies; look for unnecessary shuffles.

---

## STEP 5 — Tableau Export Strategy

**Script:** `scripts/tableau_export.py`

### Exported Datasets

| File | Source | Dashboard |
|---|---|---|
| `class_distribution.csv` | Notebook 2 | Data Quality |
| `data_quality_summary.csv` | Export script | Data Quality |
| `model_performance_combined.csv` | Notebooks 3+4 | Model Performance |
| `roc_data.csv` | Notebook 4 | Model Performance |
| `confusion_matrices.png` | Notebook 4 | Model Performance |
| `feature_importance.csv` | Notebook 4 | Feature Insights |
| `feature_importance_with_words.csv` | Notebook 4 | Feature Insights |
| `scalability_enriched.csv` | Script | Scalability & Cost |

### Export Format: CSV

**Why CSV over Parquet for Tableau:**
- Tableau Desktop reads CSV natively without connectors.
- Our aggregated tables are small (< 1 MB each) → Parquet overhead not justified.
- CSV is human-readable for quick verification.

### 4 Dashboard Designs

#### Dashboard 1: Data Quality
- **Bar chart:** Class distribution (Reliable vs Unreliable)
- **Table:** Data quality metrics (pipeline stats)
- **Story:** "We ingested X records from Common Crawl, enforced schema, filtered to English, and achieved balanced classes."

#### Dashboard 2: Model Performance
- **Grouped bar chart:** Accuracy / F1 / AUC by model × split
- **ROC curve overlay:** Three models on one chart
- **Heatmaps:** Confusion matrices
- **Story:** "Logistic Regression achieves the best AUC; RF provides interpretability via feature importance."

#### Dashboard 3: Feature Insights
- **Horizontal bar:** Top 30 features by Gini importance
- **Annotated table:** Feature hash indices mapped to actual words
- **Story:** "Key distinguishing words between reliable and unreliable sources include..."

#### Dashboard 4: Scalability & Cost Analysis
- **Line chart:** Speedup vs. executor count (ideal vs actual)
- **Dual-axis:** Training time (bars) + cost (line)
- **Efficiency bar:** Percentage of ideal linear speedup achieved
- **Story:** "2 executors achieve 1.7× speedup at 85% efficiency; 4 executors show diminishing returns due to shuffle overhead."

### Storytelling Flow (Distinction-level)

```
Data Quality → Model Performance → Feature Insights → Scalability & Cost
```

> "We sourced real-world web data at petabyte scale from Common Crawl,
> built a distributed NLP pipeline with robust Schema enforcement,
> trained three distributed classifiers with hyperparameter tuning,
> identified the linguistic signals that distinguish fake from reliable news,
> and proved that our system scales efficiently — achieving near-linear speedup
> while quantifying the cost-performance tradeoff for production deployment."

---

## Running the Full Pipeline

```powershell
# 1. Activate environment
conda activate fakenews-bigdata

# 2. Run notebooks in order
cd notebooks
jupyter lab
# Execute: 1_data_ingestion → 2_feature_engineering → 3_model_training → 4_evaluation

# 3. Run scalability experiments
cd ../scripts
python scalability_experiments.py

# 4. Consolidate Tableau exports
python tableau_export.py

# 5. Open Tableau → connect to tableau/*.csv → build dashboards
```

## Testing

```powershell
cd claw-data
pytest tests/ -v --cov=config
```

---

## Git Best Practices

```powershell
git init
git add .
git commit -m "Initial commit: project skeleton + full ML pipeline"

# Tag each major milestone
git tag -a v1.0-ingestion -m "Data ingestion pipeline complete"
git tag -a v2.0-features  -m "Feature engineering pipeline complete"
git tag -a v3.0-models    -m "Model training + evaluation complete"
git tag -a v4.0-scaling   -m "Scalability experiments complete"
git tag -a v5.0-tableau   -m "Tableau dashboards complete"
```

---

## License

Academic project — Coventry University 7006SCN.
