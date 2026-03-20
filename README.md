# Scalable Fake News Detection Using Distributed Machine Learning on Common Crawl Data


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


