# Benelux short-term-rental lakehouse

Group assignment 2 (Big Data). We built a Databricks lakehouse over Inside Airbnb data and a price model on cities A and B,
then ran them unchanged on a new client city C.

| | |
|---|---|
| **Region** | 12, Benelux (currency **EUR**, so no conversion needed) |
| **City A (training)** | Brussels, snapshot **2026-09-20** (scraped 2026-09-25), 4,850 listings |
| **City B (training)** | Antwerp, snapshot **2026-09-23** (scraped 2026-09-25/26), 2,463 listings |
| **City C (new client, unseen)** | Amsterdam, snapshot date recorded only after the week-1 freeze (see below) |
| **Why C** | Amsterdam is the largest and most expensive market in the region, in another country with its own neighbourhood system. It is the hardest realistic transfer test, compared with two mid-priced Belgian cities. |

> **Holdout discipline.** Amsterdam data was not downloaded or looked at until the pipeline and model for A and B were finished
> and committed. The commit tagged **`week1-final`** is the exact pipeline + model state used for the first city-C run.
> `git diff week1-final` shows every change made afterwards.

## Repository

```
databricks.yml, resources/jobs.yml   Databricks Asset Bundle: the city_etl job (param: city) and the lakehouse orchestrator
src/airbnb_lakehouse/                The pipeline package (single code path for every city)
    bronze.py   resolve city -> snapshot URL, land file in volume, load as strings
    silver.py   column contract, try_cast typing, cleaning, outlier flags, dq metrics
    gold.py     city / neighbourhood / room-type metrics, listing features, train/test split
    model.py    baseline, candidates, CV model choice, training, scoring, metrics
    lakehouse.py  storage: UC Delta tables on Databricks, Parquet locally
notebooks/
    00_eda_brussels_antwerp.ipynb    exploratory analysis + leakage review (cities A and B only)
    01_bronze_ingest ... 06_model_report   thin job notebooks (Databricks source format)
docs/
    architecture.md                  medallion diagram and description
    model_v1.md                      model card: CV, test metrics, error by segment, feature importance
scripts/run_local.py                 runs the same pipeline locally (PySpark local mode)
tests/                               unit tests (python -m pytest tests -q)
```

## Run it on Databricks Free Edition

```bash
databricks auth login --host https://<your-workspace>.cloud.databricks.com
databricks bundle deploy
databricks bundle run lakehouse                                   # Brussels + Antwerp: ETL -> train v1 -> score -> report
databricks bundle run city_etl --params city=antwerp              # one city, ETL only
```

If serverless compute cannot reach the internet, upload the file to
`/Volumes/workspace/airbnb_benelux/raw/<city>/<snapshot_date>/listings.csv.gz` and run the same command. Bronze falls back to the uploaded file.

## Run it locally

```bash
pip install pyspark==4.0.1 pyarrow scikit-learn pandas joblib
python scripts/run_local.py all brussels antwerp --version v1
python -m pytest tests -q
```

## Results for cities A and B (model v1, test split, EUR)

| | n | baseline MAE | model MAE | improvement |
|---|---:|---:|---:|---:|
| Brussels | 831 | 46.88 | **32.92** | 29.8 % |
| Antwerp | 402 | 57.08 | **39.01** | 31.7 % |
| Both | 1,233 | 50.21 | **34.91** | 30.5 % |

Model: gradient boosting (sklearn `HistGradientBoostingRegressor`) on log(price), chosen by 5-fold CV on the training split.
Baseline: median training price for the same room type and accommodates. Details are in [docs/model_v1.md](docs/model_v1.md).

## Data

Inside Airbnb (http://insideairbnb.com), CC BY 4.0. Only `listings.csv.gz` (detailed listings) is used.
