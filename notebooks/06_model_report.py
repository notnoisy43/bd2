# Databricks notebook source
# MAGIC %md
# MAGIC # 06 Model report
# MAGIC
# MAGIC Recomputes `gold_model_metrics` from `gold_predictions`: MAE of model and baseline per model version and city,
# MAGIC on the honest evaluation set (training cities: their 20 % test split; new cities: all priced listings), plus the pooled
# MAGIC "ALL TRAINING CITIES" row.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "airbnb_benelux")

# COMMAND ----------

# Make the shared pipeline package importable (the bundle syncs the repo to the workspace).
import os, sys
sys.path.insert(0, os.path.abspath("../src"))

from airbnb_lakehouse.pipeline import Lakehouse, run_report

lh = Lakehouse.create(spark, dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
display(run_report(lh))
