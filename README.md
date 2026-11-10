# Benelux short-term-rental lakehouse

Group assignment 2 (Big Data). We built a Databricks lakehouse over Inside Airbnb data and a price model on cities A and B,
then ran them unchanged on a new client city C.

| | |
|---|---|
| **Region** | 12, Benelux (currency **EUR**, so no conversion needed) |
| **City A (training)** | Brussels, snapshot **2026-09-20** (scraped 2026-09-25), 4,850 listings |
| **City B (training)** | Antwerp, snapshot **2026-09-23** (scraped 2026-09-25/26), 2,463 listings |
| **City C (new client, unseen)** | Amsterdam, snapshot **2026-09-16**, 10,612 listings (first downloaded by the frozen pipeline itself, after the week-1 freeze) |
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
    model_v1.md / model_v3.md        model cards: CV, test metrics, error by segment, feature importance
    city_c/report.md                 THE CITY-C REPORT: first run, every change (pipeline vs model), the gap explained
    city_c/*.log, first_run_metrics.csv   raw evidence of the first run and of each step after it
    presentation.pdf                 slides
dashboards/airbnb_benelux.lvdash.json   AI/BI dashboard (3 pages), deployed by the bundle; queries.sql = its datasets
genie/genie_space.md                 Genie space: tables, instructions, example SQL, sample questions, tricks
notebooks/07_uc_comments.sql         Unity Catalog comments used by Genie
scripts/run_local.py                 runs the same pipeline locally (PySpark local mode)
tests/                               unit tests (python -m pytest tests -q)
```

## Run it on Databricks Free Edition

```bash
databricks auth login --host https://<your-workspace>.cloud.databricks.com
databricks bundle deploy
databricks bundle run lakehouse --params target_mode=absolute    # Brussels + Antwerp: ETL -> train v1 -> score -> report
databricks bundle run city_etl --params city=antwerp              # one city, ETL only

# city C, exactly as in the first run: frozen v1, no retraining
#   (for a byte-identical first run, deploy from the tag: git checkout week1-final && databricks bundle deploy)
databricks bundle run lakehouse --params cities=amsterdam,retrain=false,model_version=v1
# after the fixes: v2 (relative price, trained on A+B) and v3 (relative price, retrained with Amsterdam)
databricks bundle run lakehouse --params cities=brussels+antwerp+amsterdam,train_cities=brussels+antwerp,model_version=v2
databricks bundle run lakehouse --params cities=brussels+antwerp+amsterdam,train_cities=brussels+antwerp+amsterdam,model_version=v3
```

Then run `notebooks/07_uc_comments.sql` once and create the Genie space with [genie/genie_space.md](genie/genie_space.md).
Model files are immutable per version: rerunning training with an existing version name fails on purpose.

If serverless compute cannot reach the internet, upload the file to
`/Volumes/workspace/airbnb_benelux/raw/<city>/<snapshot_date>/listings.csv.gz` and run the same command. Bronze falls back to the uploaded file.

## Run it locally

```bash
pip install pyspark==4.0.1 pyarrow scikit-learn pandas joblib
python scripts/run_local.py etl brussels antwerp amsterdam
python scripts/run_local.py train --version v1 --absolute                              # week-1 model type
python scripts/run_local.py train --version v2 --train-cities brussels antwerp            # relative price, A+B
python scripts/run_local.py train --version v3 --train-cities brussels antwerp amsterdam  # relative price, A+B+C
for v in v1 v2 v3; do python scripts/run_local.py score brussels antwerp amsterdam --version $v; done
python scripts/run_local.py report
python -m pytest tests -q
```

For the byte-identical first run, check out `week1-final`. Its pipeline differs from today's in the outlier rule, so a v1 retrained
on today's silver can differ marginally (one Antwerp training listing changes class). The frozen `price_model_v1.joblib` has
sha256 `74c475d6fc7d25896e7ca64b63c0512e2101fd9192278647c2b0a610d35e2a40` (recorded in the tag message).

## Results for cities A and B (model v1, frozen at `week1-final`, test split, EUR)

| | n | baseline MAE | model MAE | improvement |
|---|---:|---:|---:|---:|
| Brussels | 831 | 46.88 | **32.92** | 29.8 % |
| Antwerp | 402 | 57.08 | **39.01** | 31.7 % |
| Both | 1,233 | 50.21 | **34.91** | 30.5 % |

Model: gradient boosting (sklearn `HistGradientBoostingRegressor`) on log(price), chosen by 5-fold CV on the training split.
Baseline: median training price for the same room type and accommodates. Details are in [docs/model_v1.md](docs/model_v1.md).

## City C (Amsterdam): the short version ([full report](docs/city_c/report.md))

* **The first run worked, and that was the problem.** The unchanged pipeline resolved, downloaded and processed Amsterdam with no error and no warning,
  and the frozen v1 model produced predictions with an **MAE of €158.60** (against €34.91 on A+B). It under-priced 95 % of listings, by €155 on average,
  because Amsterdam's prices are 2.2x Belgium's and gradient-boosted trees cannot predict beyond the price range they were trained on.
* **Pipeline fixes**: an outlier ceiling relative to each city's median (the fixed €1,000 rule threw away 103 genuine Amsterdam homes);
  data-quality expectations (43 % missing prices and 42 % closed calendars now raise warnings); a scoring coverage report
  (unseen city / unseen neighbourhoods / price level).
* **Model changes**: predict price relative to the city's median (v2), drop the `city` one-hot, and retrain with Amsterdam's training split (v3).

| EUR MAE | n | |
|---|---:|---:|
| Model v1 on Brussels / Antwerp / both (test) | 831 / 402 / 1,233 | **32.92 / 39.01 / 34.91** |
| Baseline (A+B medians) on Amsterdam test | 1,160 | 210.54 |
| Baseline built on Amsterdam's own training split | 1,160 | 111.07 |
| **v1 on Amsterdam, first run** (all priced / test split) | 5,927 / 1,133 | **158.60 / 156.03** |
| v1 on Amsterdam test after the outlier-rule fix | 1,160 | 171.82 |
| **v2 on Amsterdam test** (relative price, trained on A+B only) | 1,160 | **108.75** |
| **v3 on Amsterdam test** (relative price, retrained with Amsterdam) | 1,160 | **82.89** |

Relative error: v1 on A+B 23 %, v1 on Amsterdam 45 %, v3 on Amsterdam 26 %. What is left of the euro gap is mostly the same relative error on prices that are 2.2x higher.

## Team contributions

| member | contribution |
|---|---|
| _name_ | _fill in_ |
| _name_ | _fill in_ |
| _name_ | _fill in_ |

## Data

Inside Airbnb (http://insideairbnb.com), CC BY 4.0. Only `listings.csv.gz` (detailed listings) is used.
