"""Bronze: land the raw Inside Airbnb file and load it as-is (every column a string)."""
from __future__ import annotations

import os
import re
import urllib.request

from pyspark.sql import DataFrame, functions as F

from . import config
from .lakehouse import Lakehouse

LISTINGS_FILE = "listings.csv.gz"


# --------------------------------------------------------------------------- resolve
def resolve_snapshot(city: str, snapshot_date: str = "latest", index_html: str | None = None
                     ) -> tuple[str, str]:
    """Return (snapshot_date, url) of the detailed listings file of `city`.

    `city` is the Inside Airbnb URL slug (e.g. 'brussels').  The index page lists the
    latest snapshot of every city, e.g.
    https://data.insideairbnb.com/belgium/bru/brussels/2026-09-20/data/listings.csv.gz
    """
    if index_html is None:
        req = urllib.request.Request(config.INDEX_URL, headers={"User-Agent": config.USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            index_html = resp.read().decode("utf-8", errors="replace")
    pattern = re.compile(
        r"https://data\.insideairbnb\.com/[a-z0-9\-/]+?/" + re.escape(city)
        + r"/(\d{4}-\d{2}-\d{2})/data/listings\.csv\.gz")
    found = {m.group(1): m.group(0) for m in pattern.finditer(index_html)}
    if not found:
        raise ValueError(f"City '{city}' not found on {config.INDEX_URL}")
    if snapshot_date == "latest":
        snapshot_date = max(found)
    if snapshot_date not in found:
        raise ValueError(f"Snapshot {snapshot_date} of '{city}' not listed; available: {sorted(found)}")
    return snapshot_date, found[snapshot_date]


def latest_local_snapshot(lh: Lakehouse, city: str) -> str | None:
    base = lh.raw_dir(city)
    if not os.path.isdir(base):
        return None
    dates = [d for d in os.listdir(base)
             if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) and os.path.exists(f"{base}/{d}/{LISTINGS_FILE}")]
    return max(dates) if dates else None


def land_raw_file(lh: Lakehouse, city: str, snapshot_date: str = "latest") -> tuple[str, str]:
    """Download the listings file into the raw volume (skip if already there).

    If the download is impossible (no internet on the cluster, site down) we fall back to a
    file that was uploaded manually to /Volumes/.../raw/<city>/<date>/listings.csv.gz.
    Returns (snapshot_date, path).
    """
    try:
        snapshot_date, url = resolve_snapshot(city, snapshot_date)
    except Exception as exc:  # network problem -> try the volume
        local = snapshot_date if snapshot_date != "latest" else latest_local_snapshot(lh, city)
        if local and os.path.exists(f"{lh.raw_dir(city, local)}/{LISTINGS_FILE}"):
            print(f"[bronze] could not resolve online ({exc}); using uploaded file of {local}")
            return local, f"{lh.raw_dir(city, local)}/{LISTINGS_FILE}"
        raise

    target_dir = lh.raw_dir(city, snapshot_date)
    target = f"{target_dir}/{LISTINGS_FILE}"
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print(f"[bronze] {target} already landed, skipping download")
        return snapshot_date, target
    os.makedirs(target_dir, exist_ok=True)
    tmp = target + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    os.replace(tmp, target)
    print(f"[bronze] downloaded {url} -> {target} ({os.path.getsize(target):,} bytes)")
    return snapshot_date, target


# --------------------------------------------------------------------------- load
def read_raw_listings(lh: Lakehouse, path: str) -> DataFrame:
    """Read the CSV exactly as published.  Free-text fields contain newlines and quotes,
    so multiLine + escape='"' are required; without them rows are silently split."""
    return (lh.spark.read
            .option("header", True)
            .option("multiLine", True)
            .option("escape", '"')
            .option("inferSchema", False)        # bronze keeps strings; typing is silver's job
            .option("mode", "PERMISSIVE")
            .csv(path))


def build_bronze(lh: Lakehouse, city: str, snapshot_date: str, path: str) -> DataFrame:
    df = read_raw_listings(lh, path)
    # Delta does not accept some characters in column names; normalise defensively.
    for c in df.columns:
        clean = re.sub(r"[^0-9a-zA-Z_]", "_", c.strip()).lower()
        if clean != c:
            df = df.withColumnRenamed(c, clean)
    return (df.withColumn("city", F.lit(city))
              .withColumn("snapshot_date", F.lit(snapshot_date).cast("date"))
              .withColumn("source_file", F.lit(path))
              .withColumn("ingested_at", F.current_timestamp()))


def run_bronze(lh: Lakehouse, city: str, snapshot_date: str = "latest") -> dict:
    snapshot_date, path = land_raw_file(lh, city, snapshot_date)
    df = build_bronze(lh, city, snapshot_date, path)
    if "id" not in df.columns:
        raise ValueError(f"[bronze] {path} has no 'id' column - not a listings file?")
    n = lh.write_city(df, "bronze_listings", city, merge_schema=True)
    if n == 0:
        raise ValueError(f"[bronze] {city}: 0 rows loaded from {path}")
    print(f"[bronze] {city} {snapshot_date}: {n:,} rows, {len(df.columns)} columns")
    return {"city": city, "snapshot_date": snapshot_date, "rows": n, "path": path}
