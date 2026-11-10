# Databricks notebook source
# MAGIC %md
# MAGIC # 00 Plan: turn the `cities` job parameter into the for-each list
# MAGIC
# MAGIC `cities` and `train_cities` are written as `brussels+antwerp` (any of `+ , ;` or spaces separate them).
# MAGIC The Databricks CLI splits `--params` on commas, so a JSON list cannot be passed from the command line
# MAGIC (pipeline fix P5, found while writing the run instructions in week 2).

# COMMAND ----------

import json, re

dbutils.widgets.text("cities", "brussels+antwerp")
dbutils.widgets.text("train_cities", "brussels+antwerp")
split = lambda s: [c.lower() for c in re.split(r"[+,;\s]+", s.strip()) if c]
cities, train_cities = split(dbutils.widgets.get("cities")), split(dbutils.widgets.get("train_cities"))
if not cities:
    raise ValueError("parameter 'cities' is empty")
dbutils.jobs.taskValues.set("cities", cities)
dbutils.jobs.taskValues.set("train_cities", ",".join(train_cities))
print(json.dumps({"cities": cities, "train_cities": train_cities}))
