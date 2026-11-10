"""Run the same pipeline code locally (PySpark local mode, Parquet instead of Delta).

    python scripts/run_local.py etl brussels antwerp
    python scripts/run_local.py train --version v1 --train-cities brussels antwerp
    python scripts/run_local.py score brussels antwerp --version v1
    python scripts/run_local.py report
    python scripts/run_local.py all brussels antwerp --version v1      # everything above
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession  # noqa: E402

from airbnb_lakehouse import config  # noqa: E402
from airbnb_lakehouse.pipeline import (Lakehouse, run_city_etl, run_report,  # noqa: E402
                                       run_score, run_train)


def spark_session() -> SparkSession:
    spark = (SparkSession.builder.master("local[4]").appName("airbnb-local")
             .config("spark.ui.enabled", "false")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["etl", "train", "score", "report", "all"])
    ap.add_argument("cities", nargs="*", default=config.TRAIN_CITIES)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--train-cities", nargs="+", default=config.TRAIN_CITIES)
    ap.add_argument("--snapshot", default="latest")
    ap.add_argument("--root", default=os.path.join(ROOT, "data"))
    a = ap.parse_args()

    lh = Lakehouse.create(spark_session(), local_root=a.root)
    if a.step in ("etl", "all"):
        for c in a.cities:
            run_city_etl(lh, c, a.snapshot)
    if a.step in ("train", "all"):
        run_train(lh, a.train_cities, a.version)
    if a.step in ("score", "all"):
        for c in a.cities:
            run_score(lh, c, a.version)
    if a.step in ("report", "all"):
        run_report(lh)


if __name__ == "__main__":
    main()
