# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Bronze: land and load the raw listings of one city
# MAGIC
# MAGIC **Parameter**: `city` (Inside Airbnb slug, e.g. `brussels`) and `snapshot_date` (`latest` or `YYYY-MM-DD`).
# MAGIC The notebook looks up the city's snapshot on the Inside Airbnb index, downloads `listings.csv.gz` into the
# MAGIC `raw` volume (skipped if it is already there) and loads it **as strings** into `bronze_listings`, partition `city`.
# MAGIC If the cluster has no internet access, upload the file to `/Volumes/<catalog>/<schema>/raw/<city>/<date>/listings.csv.gz` by hand.

# COMMAND ----------

dbutils.widgets.text("city", "brussels")
dbutils.widgets.text("snapshot_date", "latest")
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "airbnb_benelux")
city = dbutils.widgets.get("city").strip().lower()

# COMMAND ----------

# Make the shared pipeline package importable (the bundle syncs the repo to the workspace).
import os, sys
sys.path.insert(0, os.path.abspath("../src"))

from airbnb_lakehouse.pipeline import Lakehouse, run_bronze

lh = Lakehouse.create(spark, dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
result = run_bronze(lh, city, dbutils.widgets.get("snapshot_date").strip())
dbutils.jobs.taskValues.set("snapshot_date", result["snapshot_date"])
result

# COMMAND ----------

display(lh.read("bronze_listings").where(f"city = '{city}'").limit(5))
