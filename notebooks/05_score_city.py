# Databricks notebook source
# MAGIC %md
# MAGIC # 05 Score one city with a frozen model version
# MAGIC
# MAGIC Predictions for **every** listing of `city` (including listings without a price, which get a suggested price) go to
# MAGIC `gold_predictions` (slice `city` x `model_version`), together with the baseline prediction and the errors.

# COMMAND ----------

# MAGIC %pip install -q scikit-learn==1.6.1 joblib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("city", "brussels")
dbutils.widgets.text("model_version", "v1")
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "airbnb_benelux")
city = dbutils.widgets.get("city").strip().lower()
version = dbutils.widgets.get("model_version").strip()

# COMMAND ----------

# Make the shared pipeline package importable (the bundle syncs the repo to the workspace).
import os, sys
sys.path.insert(0, os.path.abspath("../src"))

from airbnb_lakehouse.pipeline import Lakehouse, run_score

lh = Lakehouse.create(spark, dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
run_score(lh, city, version)
