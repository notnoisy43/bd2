# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Train the price model on the training cities
# MAGIC
# MAGIC * training rows: `split = 'train'` and `price_status = 'ok'` of the training cities (default Brussels + Antwerp)
# MAGIC * model choice: 5-fold CV on the training split only (baseline, ridge, random forest, two gradient-boosting sizes)
# MAGIC * target: `log(price)`, converted back with `exp`
# MAGIC * the chosen model and the baseline are saved together, **immutably**, as `/Volumes/<catalog>/<schema>/models/price_model_<version>.joblib`
# MAGIC   (and logged to MLflow). Scoring later loads exactly this file, which is how the city-C first run uses the unchanged week-1 model.

# COMMAND ----------

# MAGIC %pip install -q scikit-learn==1.6.1 joblib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("train_cities", "brussels,antwerp")
dbutils.widgets.text("model_version", "v1")
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "airbnb_benelux")
train_cities = [c.strip().lower() for c in dbutils.widgets.get("train_cities").split(",") if c.strip()]
version = dbutils.widgets.get("model_version").strip()

# COMMAND ----------

# Make the shared pipeline package importable (the bundle syncs the repo to the workspace).
import os, sys
sys.path.insert(0, os.path.abspath("../src"))

from airbnb_lakehouse.pipeline import Lakehouse, run_train

lh = Lakehouse.create(spark, dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
run_train(lh, train_cities, version)
