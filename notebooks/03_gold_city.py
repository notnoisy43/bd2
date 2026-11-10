# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Gold: city metrics and model features
# MAGIC
# MAGIC Writes `gold_listing_features` (one row per listing, leakage-free features, deterministic train/test split),
# MAGIC `gold_city_metrics`, `gold_neighbourhood_metrics` and `gold_room_type_metrics` for one city.

# COMMAND ----------

dbutils.widgets.text("city", "brussels")
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "airbnb_benelux")
city = dbutils.widgets.get("city").strip().lower()

# COMMAND ----------

# Make the shared pipeline package importable (the bundle syncs the repo to the workspace).
import os, sys
sys.path.insert(0, os.path.abspath("../src"))

from airbnb_lakehouse.pipeline import Lakehouse, run_gold

lh = Lakehouse.create(spark, dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
run_gold(lh, city)

# COMMAND ----------

display(lh.read("gold_city_metrics").where(f"city = '{city}'"))
