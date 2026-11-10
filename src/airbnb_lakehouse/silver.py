"""Silver: typed, cleaned, de-duplicated listings with data-quality flags.

Design rules
* An explicit column contract.  Required columns missing -> fail loudly.  Optional columns
  missing -> created as NULL with a warning (Inside Airbnb adds/empties columns between
  snapshots; e.g. `host_since`, `bathrooms` and `license` are 100 % empty in the 2026-09
  Belgian snapshots).
* All casts are `try_cast`: Databricks serverless runs with ANSI SQL on, where a single
  malformed value would otherwise fail the whole job.
* Rows are never silently dropped: each drop reason is counted into `silver_dq`.
* Price outliers are flagged (price_status), not deleted: the dashboard still shows them.
* Columns computed from the price (price_quote_*) are not carried into silver at all.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, Window, functions as F

from . import config
from .lakehouse import Lakehouse

REQUIRED = ["id", "price", "room_type", "accommodates", "latitude", "longitude",
            "neighbourhood_cleansed"]

TYPES = {
    # identifiers / text
    "id": "bigint", "host_id": "bigint",
    "listing_url": "string", "name": "string", "host_name": "string",
    "neighbourhood_cleansed": "string", "property_type": "string", "room_type": "string",
    "bathrooms_text": "string", "license": "string",
    # numerics
    "latitude": "double", "longitude": "double", "bathrooms": "double", "bedrooms": "double",
    "beds": "double", "accommodates": "int",
    "minimum_nights": "int", "maximum_nights": "int",
    "availability_30": "int", "availability_60": "int", "availability_90": "int",
    "availability_365": "int",
    "number_of_reviews": "int", "number_of_reviews_ltm": "int", "number_of_reviews_l30d": "int",
    "reviews_per_month": "double",
    "review_scores_rating": "double", "review_scores_accuracy": "double",
    "review_scores_cleanliness": "double", "review_scores_checkin": "double",
    "review_scores_communication": "double", "review_scores_location": "double",
    "review_scores_value": "double",
    "host_listings_count": "int", "calculated_host_listings_count": "int",
    "calculated_host_listings_count_entire_homes": "int",
    "hosts_time_as_host_years": "double",
    "estimated_occupancy_l365d": "int",
    "estimated_revenue_l365d": "double",   # = price x occupancy -> dashboard only, never a feature
    # booleans ('t'/'f')
    "host_is_superhost": "boolean", "host_identity_verified": "boolean",
    "has_availability": "boolean", "instant_bookable": "boolean",
    # dates
    "last_scraped": "date", "host_since": "date", "first_review": "date", "last_review": "date",
    # special
    "amenities": "array<string>", "host_response_rate": "pct", "host_acceptance_rate": "pct",
}


def _cast(col: str, typ: str) -> F.Column:
    c = F.col(f"`{col}`")
    if typ == "boolean":
        return (F.when(F.lower(F.trim(c)).isin("t", "true"), F.lit(True))
                 .when(F.lower(F.trim(c)).isin("f", "false"), F.lit(False)))
    if typ in ("int", "bigint"):
        # via double: tolerates "4.0"; try_cast: tolerates garbage under ANSI mode
        return F.expr(f"try_cast(try_cast(`{col}` AS double) AS {typ})")
    if typ == "pct":
        return F.expr(f"try_cast(regexp_replace(`{col}`, '%', '') AS double) / 100")
    if typ == "array<string>":
        return F.from_json(c, "array<string>")
    if typ == "string":
        return F.when(F.trim(c) != "", F.trim(c))
    return F.expr(f"try_cast(`{col}` AS {typ})")


def parse_price(col: str = "price") -> F.Column:
    """'$1,234.00' -> 1234.0 (Inside Airbnb prints '$' whatever the local currency)."""
    return F.expr(f"try_cast(regexp_replace(`{col}`, '[^0-9.]', '') AS double)")


def conform_schema(bronze: DataFrame) -> tuple[DataFrame, list[str]]:
    missing_required = [c for c in REQUIRED if c not in bronze.columns]
    if missing_required:
        raise ValueError(f"[silver] required columns missing from bronze: {missing_required}")
    added = []
    for col in TYPES:
        if col not in bronze.columns:
            bronze = bronze.withColumn(col, F.lit(None).cast("string"))
            added.append(col)
    if added:
        print(f"[silver] WARNING optional columns absent in this snapshot, filled with NULL: {added}")
    return bronze, added


def build_silver(bronze: DataFrame) -> tuple[DataFrame, dict]:
    raw, added = conform_schema(bronze)
    n_bronze = raw.count()

    typed = raw.select(
        "city", "snapshot_date",
        *[_cast(c, t).alias(c) for c, t in TYPES.items() if c != "price"],
        F.col("price").alias("price_raw"),
        parse_price("price").alias("price"),
    )

    # --- derived, row-level ---------------------------------------------------------------
    bt = F.lower(F.col("bathrooms_text"))
    bath_from_text = (F.when(bt.contains("half-bath") & ~bt.rlike(r"^\s*[0-9]"), F.lit(0.5))
                       .otherwise(F.expr("try_cast(regexp_extract(lower(bathrooms_text), '^([0-9.]+)', 1) AS double)")))
    typed = (typed
             .withColumn("bathrooms", F.coalesce("bathrooms", bath_from_text))
             .withColumn("bathroom_shared", bt.contains("shared"))
             .withColumn("property_base", F.lower(F.trim(F.regexp_replace(
                 "property_type", r"(?i)^(entire|private room in|shared room in|room in)\s+", ""))))
             .withColumn("amenities_count", F.when(F.col("amenities").isNotNull(), F.size("amenities")))
             # 25 bedrooms for 2 guests etc. -> value is not credible, keep row, null the value
             .withColumn("bedrooms", F.when(F.col("bedrooms") <= 15, F.col("bedrooms")))
             .withColumn("beds", F.when(F.col("beds") <= 30, F.col("beds")))
             .withColumn("is_long_stay", F.col("minimum_nights") >= config.LONG_STAY_MIN_NIGHTS)
             .withColumn("is_active", (F.col("availability_365") > 0) | (F.col("number_of_reviews_ltm") > 0))
             .withColumn("is_multi_listing_host", F.col("calculated_host_listings_count") > 1)
             .withColumn("price_status",
                         F.when(F.col("price").isNull(), "missing")
                          .when(F.col("price") < config.PRICE_LOW, "low_outlier")
                          .when(F.col("price") > config.PRICE_HIGH, "high_outlier")
                          .otherwise("ok")))

    # --- row filters (counted) -------------------------------------------------------------
    no_id = typed.where(F.col("id").isNull()).count()
    typed = typed.where(F.col("id").isNotNull())
    w = Window.partitionBy("id").orderBy(F.col("last_scraped").desc_nulls_last())
    deduped = typed.withColumn("_rn", F.row_number().over(w)).where("_rn = 1").drop("_rn")
    n_after_dedup = deduped.count()
    core_ok = (F.col("latitude").isNotNull() & F.col("longitude").isNotNull()
               & F.col("room_type").isNotNull() & F.col("accommodates").isNotNull()
               & (F.col("accommodates") > 0))
    no_core = deduped.where(~core_ok).count()
    silver = deduped.where(core_ok)

    status = {r["price_status"]: r["count"] for r in
              silver.groupBy("price_status").count().collect()}
    dq = {
        "bronze_rows": n_bronze,
        "dropped_missing_id": no_id,
        "dropped_duplicate_id": n_bronze - no_id - n_after_dedup,
        "dropped_missing_core_fields": no_core,
        "silver_rows": n_after_dedup - no_core,
        "optional_columns_missing": len(added),
        **{f"price_status_{k}": v for k, v in status.items()},
    }
    return silver, dq


def dq_frame(lh: Lakehouse, city: str, layer: str, dq: dict) -> DataFrame:
    import pandas as pd
    rows = pd.DataFrame({"city": city, "layer": layer, "metric": list(dq.keys()),
                         "value": [float(v) for v in dq.values()]})
    return lh.spark.createDataFrame(rows).withColumn("computed_at", F.current_timestamp())


def run_silver(lh: Lakehouse, city: str) -> dict:
    bronze = lh.read("bronze_listings").where(F.col("city") == city)
    silver, dq = build_silver(bronze)
    n = lh.write_city(silver, "silver_listings", city)
    lh.write_city(dq_frame(lh, city, "silver", dq), "dq_metrics", city)
    print(f"[silver] {city}: {n:,} rows | " + ", ".join(f"{k}={v}" for k, v in dq.items()))
    return {"city": city, **dq}
