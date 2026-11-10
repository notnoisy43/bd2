"""Unit tests for the pure parts of the pipeline (run: python -m pytest tests -q)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

from airbnb_lakehouse.bronze import resolve_snapshot  # noqa: E402
from airbnb_lakehouse.model import MedianBaseline, metrics_table  # noqa: E402

FAKE_INDEX = """
<a href="https://data.insideairbnb.com/belgium/bru/brussels/2026-09-20/data/listings.csv.gz">x</a>
<a href="https://data.insideairbnb.com/belgium/bru/brussels/2026-06-18/data/listings.csv.gz">x</a>
<a href="https://data.insideairbnb.com/belgium/vlg/antwerp/2026-09-23/data/listings.csv.gz">x</a>
"""


def test_resolve_latest_and_pinned():
    assert resolve_snapshot("brussels", "latest", FAKE_INDEX)[0] == "2026-09-20"
    d, url = resolve_snapshot("brussels", "2026-06-18", FAKE_INDEX)
    assert url.endswith("/brussels/2026-06-18/data/listings.csv.gz")
    with pytest.raises(ValueError):
        resolve_snapshot("ghent", "latest", FAKE_INDEX)


def test_baseline_fallbacks():
    df = pd.DataFrame({"room_type": ["A", "A", "A", "B"], "accommodates": [2, 2, 4, 2]})
    b = MedianBaseline().fit(df, pd.Series([100, 120, 200, 50]))
    q = pd.DataFrame({"room_type": ["A", "A", "B", "Z"], "accommodates": [2, 3, 2, 2]})
    np.testing.assert_allclose(b.predict(q), [110, 120, 50, 110])   # exact, room-type, exact, global


def test_metrics_table_eval_sets():
    p = pd.DataFrame({
        "model_version": "v1", "city": ["a", "a", "c", "c"],
        "city_role": ["training_city", "training_city", "new_city", "new_city"],
        "split": ["train", "test", "train", "test"], "price_status": "ok",
        "price": [100.0, 100.0, 100.0, 200.0], "predicted_price": [0.0, 90.0, 150.0, 150.0],
        "baseline_price": [100.0, 80.0, 100.0, 100.0]})
    p["error"] = p.predicted_price - p.price
    p["abs_error"] = p.error.abs()
    m = metrics_table(p).set_index("city")
    assert m.loc["a", "n"] == 1 and m.loc["a", "mae_model"] == 10      # train rows never evaluated
    assert m.loc["c", "n"] == 2 and m.loc["c", "mae_model"] == 50      # new city: all rows
    assert "ALL TRAINING CITIES" in m.index


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession
    s = (SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true").getOrCreate())
    yield s


def test_silver_parsing(spark):
    from airbnb_lakehouse.silver import TYPES, build_silver
    cols = ["id", "price", "room_type", "accommodates", "latitude", "longitude",
            "neighbourhood_cleansed", "bathrooms_text", "host_is_superhost", "minimum_nights",
            "amenities", "bedrooms"]
    rows = pd.DataFrame([
        ["1", "$1,234.00", "Entire home/apt", "4", "50.8", "4.3", "X", "1.5 shared baths", "t", "30", '["Wifi","Dishwasher"]', "2"],
        ["1", "$99.00", "Entire home/apt", "4", "50.8", "4.3", "X", "1 bath", "t", "1", "[]", "2"],  # duplicate id
        ["2", "", "Private room", "2.0", "50.8", "4.3", "X", "Half-bath", "f", "2", "not json", "25"],
        ["3", "$50.00", "Private room", "abc", "50.8", "4.3", "X", "1 bath", "f", "2", "[]", "1"],  # bad accommodates
    ], columns=cols)
    from pyspark.sql import functions as F
    bronze = (spark.createDataFrame(rows).withColumn("city", F.lit("t"))
              .withColumn("snapshot_date", F.lit("2026-01-01").cast("date")))
    silver, dq = build_silver(bronze)
    out = silver.toPandas().set_index("id")
    assert dq["optional_columns_missing"] == len([c for c in TYPES if c not in cols])
    assert dq["dropped_duplicate_id"] == 1 and dq["dropped_missing_core_fields"] == 1
    assert set(out.index) == {1, 2}
    assert out.loc[1, "price"] in (1234.0, 99.0) and out.loc[2, "price_status"] == "missing"
    assert out.loc[1, "amenities_count"] in (0, 2) and pd.isna(out.loc[2, "amenities_count"])  # bad JSON -> NULL
    assert out.loc[2, "bathrooms"] == 0.5 and pd.isna(out.loc[2, "bedrooms"])      # half-bath; 25 bedrooms -> NULL
    assert out.loc[2, "accommodates"] == 2                                          # "2.0" -> 2
