# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Silver: typed, cleaned, de-duplicated listings
# MAGIC
# MAGIC Column contract, `try_cast` typing, price parsing, bathrooms from text, amenities array, outlier flags
# MAGIC (`price_status`), de-duplication. Every dropped row is counted in `dq_metrics`. The rules and their reasons
# MAGIC are in `00_eda_brussels_antwerp`, section 5.

# COMMAND ----------

dbutils.widgets.text("city", "brussels")
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "airbnb_benelux")
city = dbutils.widgets.get("city").strip().lower()

# COMMAND ----------

# Make the shared pipeline package importable (the bundle syncs the repo to the workspace).
import os, sys
sys.path.insert(0, os.path.abspath("../src"))

from airbnb_lakehouse.pipeline import Lakehouse, run_silver

lh = Lakehouse.create(spark, dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
run_silver(lh, city)

# COMMAND ----------

display(lh.read("dq_metrics").where(f"city = '{city}'"))
