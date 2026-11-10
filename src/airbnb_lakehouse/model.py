"""Price model: baseline, candidates, training, scoring and evaluation.

The modelling itself is scikit-learn on pandas (a few thousand rows - Spark ML would add
nothing but latency).  Reading features and writing predictions goes through Spark/Delta.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import joblib
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from . import config
from .lakehouse import Lakehouse

# ------------------------------------------------------------------------------- features
NUMERIC = [
    "accommodates", "bedrooms", "beds", "bathrooms", "bathroom_shared", "amenities_count",
    "has_air_conditioning", "has_dishwasher", "has_washer", "has_free_parking", "has_elevator",
    "has_self_checkin", "has_workspace", "has_view", "has_hot_tub_or_pool",
    "minimum_nights", "is_long_stay",
    "number_of_reviews", "number_of_reviews_ltm", "reviews_per_month",
    "review_scores_rating", "review_scores_cleanliness", "review_scores_location",
    "review_scores_value",
    "host_is_superhost", "calculated_host_listings_count", "hosts_time_as_host_years",
    "dist_centre_km",
]
CATEGORICAL = ["room_type", "property_group", "city"]
TARGET_ENCODED = ["nbhd_key"]           # city|neighbourhood, smoothed target encoding
FEATURES = NUMERIC + CATEGORICAL + TARGET_ENCODED

# property types seen >= ~10 times in A+B; everything else -> 'other'
PROPERTY_GROUPS = ["rental unit", "home", "condo", "loft", "townhouse", "serviced apartment",
                   "hotel", "boutique hotel", "aparthotel", "bed and breakfast", "guesthouse",
                   "villa", "guest suite"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model columns from a gold_listing_features frame (pure pandas)."""
    out = df.copy()
    out["property_group"] = out["property_base"].where(out["property_base"].isin(PROPERTY_GROUPS), "other")
    out["nbhd_key"] = out["city"].astype(str) + "|" + out["neighbourhood_cleansed"].astype(str)
    for c in NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


# ------------------------------------------------------------------------------- baseline
class MedianBaseline:
    """Median training price for the same (room_type, accommodates); falls back to the
    room_type median, then to the global median when a combination was never seen."""

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "MedianBaseline":
        d = df[["room_type", "accommodates"]].assign(y=y.values)
        self.by_rt_acc = d.groupby(["room_type", "accommodates"])["y"].median().to_dict()
        self.by_rt = d.groupby("room_type")["y"].median().to_dict()
        self.global_ = float(d["y"].median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([
            self.by_rt_acc.get((rt, acc), self.by_rt.get(rt, self.global_))
            for rt, acc in zip(df["room_type"], df["accommodates"])], dtype=float)


# ------------------------------------------------------------------------------- candidates
def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), CATEGORICAL),
        ("te", TargetEncoder(target_type="continuous", random_state=config.SPLIT_SEED), TARGET_ENCODED),
    ], remainder="drop", verbose_feature_names_out=False)


def candidates() -> dict:
    """All candidates predict log(price) and are converted back with exp."""
    def wrap(pre, est):
        return TransformedTargetRegressor(regressor=Pipeline([("pre", pre), ("est", est)]),
                                          func=np.log, inverse_func=np.exp)
    return {
        "ridge": wrap(_preprocessor(True), Ridge(alpha=3.0)),
        "random_forest": wrap(_preprocessor(False), RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.5, n_jobs=-1,
            random_state=config.SPLIT_SEED)),
        "hgb_small": wrap(_preprocessor(False), HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=1.0, random_state=config.SPLIT_SEED)),
        "hgb": wrap(_preprocessor(False), HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=20,
            l2_regularization=1.0, random_state=config.SPLIT_SEED)),
    }


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def cross_validate(train: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Model choice on the training split only (the test split is never touched here)."""
    X, y = train[FEATURES], train["price"]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.SPLIT_SEED)
    rows = []
    for fold, (tr, va) in enumerate(kf.split(X)):
        base = MedianBaseline().fit(X.iloc[tr], y.iloc[tr])
        rows.append({"model": "baseline", "fold": fold, "mae": mae(y.iloc[va], base.predict(X.iloc[va]))})
        for name, est in candidates().items():
            est.fit(X.iloc[tr], y.iloc[tr])
            rows.append({"model": name, "fold": fold, "mae": mae(y.iloc[va], est.predict(X.iloc[va]))})
    cv = pd.DataFrame(rows).groupby("model")["mae"].agg(["mean", "std"]).sort_values("mean")
    return cv


# ------------------------------------------------------------------------------- train
def model_path(lh: Lakehouse, version: str) -> str:
    return f"{lh.model_dir()}/price_model_{version}.joblib"


def train_model(features: pd.DataFrame, train_cities: list[str], version: str) -> dict:
    df = prepare(features[features["city"].isin(train_cities)])
    train = df[(df["split"] == "train") & (df["price_status"] == "ok")].reset_index(drop=True)
    print(f"[train] {len(train):,} training rows from {train_cities}")
    cv = cross_validate(train)
    print("[train] 5-fold CV MAE on the training split (EUR):\n" + cv.round(2).to_string())
    best = next(m for m in cv.index if m != "baseline")
    model = candidates()[best].fit(train[FEATURES], train["price"])
    baseline = MedianBaseline().fit(train[FEATURES], train["price"])
    return {
        "version": version, "model_name": best, "model": model, "baseline": baseline,
        "features": FEATURES, "train_cities": list(train_cities), "n_train": len(train),
        "cv": cv.reset_index().to_dict(orient="records"),
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "price_bounds": [config.PRICE_LOW, config.PRICE_HIGH],
    }


def save_bundle(lh: Lakehouse, bundle: dict) -> str:
    path = model_path(lh, bundle["version"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists - model versions are immutable, choose a new version")
    joblib.dump(bundle, path)
    meta = {k: v for k, v in bundle.items() if k not in ("model", "baseline")}
    with open(path.replace(".joblib", ".json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path


def load_bundle(lh: Lakehouse, version: str) -> dict:
    return joblib.load(model_path(lh, version))


def log_to_mlflow(bundle: dict, path: str) -> None:
    """Best effort: on Databricks log params/CV metrics/artifact to MLflow; skip locally."""
    try:
        from .lakehouse import is_databricks
        if not is_databricks():
            return
        import mlflow
        with mlflow.start_run(run_name=f"price_model_{bundle['version']}"):
            mlflow.log_params({"model": bundle["model_name"], "version": bundle["version"],
                               "train_cities": ",".join(bundle["train_cities"]),
                               "n_train": bundle["n_train"]})
            for r in bundle["cv"]:
                mlflow.log_metric(f"cv_mae_{r['model']}", r["mean"])
            mlflow.log_artifact(path)
    except Exception as exc:  # MLflow must never break the pipeline
        print(f"[train] MLflow logging skipped: {exc}")


def run_train(lh: Lakehouse, train_cities: list[str], version: str) -> dict:
    feats = (lh.read("gold_listing_features").where(F.col("city").isin(train_cities)).toPandas())
    bundle = train_model(feats, train_cities, version)
    path = save_bundle(lh, bundle)
    log_to_mlflow(bundle, path)
    print(f"[train] saved {bundle['model_name']} as {path}")
    return {"version": version, "model": bundle["model_name"], "path": path, "n_train": bundle["n_train"]}


# ------------------------------------------------------------------------------- score
def predict_frame(bundle: dict, features: pd.DataFrame) -> pd.DataFrame:
    df = prepare(features)
    pred = bundle["model"].predict(df[bundle["features"]])
    base = bundle["baseline"].predict(df)
    training_city = df["city"].isin(bundle["train_cities"])
    out = pd.DataFrame({
        "city": df["city"], "listing_id": df["listing_id"].astype("int64"),
        "model_version": bundle["version"], "model_name": bundle["model_name"],
        "city_role": np.where(training_city, "training_city", "new_city"),
        "split": df["split"], "price": df["price"], "price_status": df["price_status"],
        "predicted_price": np.round(pred, 2), "baseline_price": np.round(base, 2),
        "room_type": df["room_type"], "accommodates": df["accommodates"],
        "property_group": df["property_group"], "neighbourhood": df["neighbourhood_cleansed"],
        "name": df["name"], "listing_url": df["listing_url"],
        "latitude": df["latitude"], "longitude": df["longitude"],
    })
    out["error"] = np.round(out["predicted_price"] - out["price"], 2)          # >0 over-priced
    out["abs_error"] = out["error"].abs()
    out["baseline_abs_error"] = np.round((out["baseline_price"] - out["price"]).abs(), 2)
    out["pct_error"] = np.round(out["error"] / out["price"], 4)
    return out


def run_score(lh: Lakehouse, city: str, version: str) -> dict:
    bundle = load_bundle(lh, version)
    feats = lh.read("gold_listing_features").where(F.col("city") == city).toPandas()
    if feats.empty:
        raise ValueError(f"[score] no gold features for {city}")
    preds = predict_frame(bundle, feats)
    sdf = (lh.spark.createDataFrame(preds)
           .withColumn("scored_at", F.current_timestamp()))
    n = lh.write_city(sdf, "gold_predictions", city, extra={"model_version": version})
    ok = preds[preds["price_status"] == "ok"]
    ev = ok[ok["split"] == "test"] if city in bundle["train_cities"] else ok
    summary = {"city": city, "version": version, "rows": n, "eval_rows": len(ev),
               "mae_model": round(mae(ev["price"], ev["predicted_price"]), 2) if len(ev) else None,
               "mae_baseline": round(mae(ev["price"], ev["baseline_price"]), 2) if len(ev) else None}
    print(f"[score] {summary}")
    return summary


# ------------------------------------------------------------------------------- report
def metrics_table(preds: pd.DataFrame) -> pd.DataFrame:
    """MAE etc. per (model_version, city) on the honest evaluation set:
    training cities -> their 20 % test split; new cities -> every priced listing
    (the model never saw them).  Plus one pooled row for all training cities."""
    p = preds[preds["price_status"] == "ok"].copy()
    p["eval"] = (p["city_role"] == "new_city") | (p["split"] == "test")
    p = p[p["eval"]]
    groups = []
    for (ver, city, role), g in p.groupby(["model_version", "city", "city_role"]):
        groups.append((ver, city, role, "test split" if role == "training_city" else "all listings", g))
    for ver, g in p[p["city_role"] == "training_city"].groupby("model_version"):
        groups.append((ver, "ALL TRAINING CITIES", "training_city", "test split", g))
    rows = []
    for ver, city, role, ev, g in groups:
        rows.append({
            "model_version": ver, "city": city, "city_role": role, "evaluation_set": ev,
            "n": len(g),
            "mae_model": round(mae(g["price"], g["predicted_price"]), 2),
            "mae_baseline": round(mae(g["price"], g["baseline_price"]), 2),
            "median_abs_error_model": round(float(g["abs_error"].median()), 2),
            "mape_model": round(float((g["abs_error"] / g["price"]).mean()), 4),
            "mean_error_model": round(float(g["error"].mean()), 2),
            "median_price": round(float(g["price"].median()), 2),
        })
    m = pd.DataFrame(rows)
    m["improvement_vs_baseline"] = np.round(1 - m["mae_model"] / m["mae_baseline"], 4)
    return m


def run_report(lh: Lakehouse) -> pd.DataFrame:
    preds = lh.read("gold_predictions").toPandas()
    m = metrics_table(preds)
    lh.write_full(lh.spark.createDataFrame(m).withColumn("computed_at", F.current_timestamp()),
                  "gold_model_metrics")
    print("[report]\n" + m.to_string(index=False))
    return m
