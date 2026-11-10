"""Entry points shared by the Databricks notebooks and the local runner."""
from __future__ import annotations

from .bronze import run_bronze
from .gold import run_gold
from .lakehouse import Lakehouse
from .model import run_report, run_score, run_train
from .silver import run_silver


def run_city_etl(lh: Lakehouse, city: str, snapshot_date: str = "latest") -> dict:
    """bronze -> silver -> gold for one city.  The only input is the city slug."""
    b = run_bronze(lh, city, snapshot_date)
    s = run_silver(lh, city)
    g = run_gold(lh, city)
    return {"bronze": b, "silver": s, "gold": g}


__all__ = ["Lakehouse", "run_bronze", "run_silver", "run_gold", "run_city_etl",
           "run_train", "run_score", "run_report"]
