"""Storage abstraction.

On Databricks every table is a Unity Catalog Delta table `<catalog>.<schema>.<name>`
partitioned by `city`, and every per-city write overwrites only that city's partition
(`replaceWhere`).  That makes each city run idempotent and lets cities run in parallel.

Locally (for development and tests, no Databricks needed) the same calls read/write
Parquet under `data/lakehouse/<name>/city=<city>/`.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession

from . import config


def is_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ or os.path.isdir("/Workspace")


@dataclass
class Lakehouse:
    spark: SparkSession
    catalog: str = config.DEFAULT_CATALOG
    schema: str = config.DEFAULT_SCHEMA
    local_root: str | None = None  # set => local mode

    # ------------------------------------------------------------------ factory
    @classmethod
    def create(cls, spark: SparkSession, catalog: str | None = None, schema: str | None = None,
               local_root: str | None = None) -> "Lakehouse":
        lh = cls(spark, catalog or config.DEFAULT_CATALOG, schema or config.DEFAULT_SCHEMA,
                 None if is_databricks() else (local_root or "data"))
        lh.ensure_namespace()
        return lh

    @property
    def local(self) -> bool:
        return self.local_root is not None

    def ensure_namespace(self) -> None:
        if self.local:
            for d in ("raw", "lakehouse", "models"):
                os.makedirs(os.path.join(self.local_root, d), exist_ok=True)
            return
        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.catalog}.{self.schema}")
        for vol in (config.RAW_VOLUME, config.MODEL_VOLUME):
            self.spark.sql(f"CREATE VOLUME IF NOT EXISTS {self.catalog}.{self.schema}.{vol}")

    # ------------------------------------------------------------------ paths
    def raw_dir(self, city: str, snapshot_date: str | None = None) -> str:
        base = (os.path.join(self.local_root, "raw") if self.local
                else f"/Volumes/{self.catalog}/{self.schema}/{config.RAW_VOLUME}")
        parts = [base, city] + ([snapshot_date] if snapshot_date else [])
        return "/".join(p.replace("\\", "/") for p in parts)

    def model_dir(self) -> str:
        return (os.path.join(self.local_root, "models").replace("\\", "/") if self.local
                else f"/Volumes/{self.catalog}/{self.schema}/{config.MODEL_VOLUME}")

    def fq(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    def _local_path(self, name: str) -> str:
        return os.path.join(self.local_root, "lakehouse", name).replace("\\", "/")

    # ------------------------------------------------------------------ read
    def exists(self, name: str) -> bool:
        if self.local:
            p = self._local_path(name)
            return os.path.isdir(p) and any(f.startswith("city=") or f.endswith(".parquet")
                                            for f in os.listdir(p))
        return self.spark.catalog.tableExists(self.fq(name))

    def read(self, name: str) -> DataFrame:
        if self.local:
            return self._read_local(self._local_path(name))
        return self.spark.table(self.fq(name))

    # ------------------------------------------------------------------ write
    def write_city(self, df: DataFrame, name: str, city: str, merge_schema: bool = False,
                   extra: dict | None = None) -> int:
        """Overwrite one city's slice of `name` (optionally narrowed by `extra` key=value
        pairs, e.g. model_version); other slices are untouched."""
        keys = {"city": city, **(extra or {})}
        predicate = " AND ".join(f"{k} = '{v}'" for k, v in keys.items())
        if self.local:
            path = os.path.join(self._local_path(name), *[f"{k}={v}" for k, v in keys.items()])
            self._write_local(df.drop(*keys.keys()), path)
        else:
            writer = df.write.format("delta").mode("overwrite")
            if merge_schema:
                writer = writer.option("mergeSchema", "true")
            if self.exists(name):
                writer.option("replaceWhere", predicate).saveAsTable(self.fq(name))
            else:
                writer.partitionBy("city").saveAsTable(self.fq(name))
        return self.read(name).where(predicate).count()

    def write_full(self, df: DataFrame, name: str) -> None:
        """Overwrite a whole (small, non-city) table."""
        if self.local:
            self._write_local(df, self._local_path(name))
        else:
            (df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
               .saveAsTable(self.fq(name)))

    def _read_local(self, path: str) -> DataFrame:
        # Spark's Hadoop file listing needs winutils on Windows, so local mode reads with
        # Arrow: one file per partition dir, partition keys re-added from the k=v path,
        # schemas unified (bronze columns can differ between cities).
        import pyarrow as pa
        import pyarrow.parquet as pq
        tables = []
        for dirpath, _, files in os.walk(path):
            for f in files:
                if not f.endswith(".parquet"):
                    continue
                t = pq.read_table(os.path.join(dirpath, f))
                rel = os.path.relpath(dirpath, path)
                for part in ([] if rel == "." else rel.replace("\\", "/").split("/")):
                    k, v = part.split("=", 1)
                    t = t.append_column(k, pa.array([v] * t.num_rows, pa.string()))
                tables.append(t)
        if not tables:
            raise FileNotFoundError(path)
        return self.spark.createDataFrame(pa.concat_tables(tables, promote_options="default"))

    @staticmethod
    def _write_local(df: DataFrame, path: str) -> None:
        # Spark's own writer needs winutils on Windows; Arrow is enough for local dev.
        import pyarrow.parquet as pq
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        pq.write_table(df.toArrow(), os.path.join(path, "part-0.parquet"))
