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

# [city-C model change M2] the one-hot `city` feature carries no information for a city the
# model never saw (all-zero vector = "neither Brussels nor Antwerp").  Relative models drop it;
# the city enters only through its price index (M1).
CATEGORICAL_REL = ["room_type", "property_group"]
FEATURES_REL = NUMERIC + CATEGORICAL_REL + TARGET_ENCODED

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
def _preprocessor(scale_numeric: bool, categorical: list[str] = CATEGORICAL) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), categorical),
        ("te", TargetEncoder(target_type="continuous", random_state=config.SPLIT_SEED), TARGET_ENCODED),
    ], remainder="drop", verbose_feature_names_out=False)


def candidates(categorical: list[str] = CATEGORICAL) -> dict:
    """All candidates predict log(target) and are converted back with exp."""
    def wrap(pre, est):
        return TransformedTargetRegressor(regressor=Pipeline([("pre", pre), ("est", est)]),
                                          func=np.log, inverse_func=np.exp)
    return {
        "ridge": wrap(_preprocessor(True, categorical), Ridge(alpha=3.0)),
        "random_forest": wrap(_preprocessor(False, categorical), RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.5, n_jobs=-1,
            random_state=config.SPLIT_SEED)),
        "hgb_small": wrap(_preprocessor(False, categorical), HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=1.0, random_state=config.SPLIT_SEED)),
        "hgb": wrap(_preprocessor(False, categorical), HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=20,
            l2_regularization=1.0, random_state=config.SPLIT_SEED)),
    }


# [city-C model change M1] price level.  v1 learned absolute EUR prices of two Belgian cities;
# trees cannot predict above what they saw, so in Amsterdam (median 2.2x) 95 % of listings were
# under-priced.  A relative model learns price / city_index, where city_index is the median
# price of the city's priced *training-split* listings, and multiplies back at prediction.
# For a new city the index comes from that city's own training split: one number, no test labels.
def city_index(df: pd.DataFrame) -> dict:
    ok = df[(df["split"] == "train") & (df["price_status"] == "ok")]
    return ok.groupby("city")["price"].median().astype(float).to_dict()


class RelativePriceModel:
    """Wraps an estimator that predicts log(price / city_index)."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.Series, index: dict) -> "RelativePriceModel":
        self.index_ = dict(index)
        self.estimator.fit(X, y.values / X["city"].map(self.index_).values)
        return self

    def predict(self, X: pd.DataFrame, index: dict | None = None) -> np.ndarray:
        idx = {**(index or {}), **self.index_}
        level = X["city"].map(idx)
        if level.isna().any():
            raise ValueError(f"no price index for cities {sorted(X.loc[level.isna(), 'city'].unique())}")
        return self.estimator.predict(X) * level.values


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def cross_validate(train: pd.DataFrame, n_splits: int = 5, relative: bool = False) -> pd.DataFrame:
    """Model choice on the training split only (the test split is never touched here)."""
    feats = (FEATURES_REL + ["city"]) if relative else FEATURES
    X, y = train[feats], train["price"]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.SPLIT_SEED)
    rows = []
    for fold, (tr, va) in enumerate(kf.split(X)):
        base = MedianBaseline().fit(X.iloc[tr], y.iloc[tr])
        rows.append({"model": "baseline", "fold": fold, "mae": mae(y.iloc[va], base.predict(X.iloc[va]))})
        for name, est in candidates(CATEGORICAL_REL if relative else CATEGORICAL).items():
            if relative:   # city index from the fold's training rows only
                idx = train.iloc[tr].groupby("city")["price"].median().to_dict()
                m = RelativePriceModel(est).fit(X.iloc[tr], y.iloc[tr], idx)
            else:
                m = est.fit(X.iloc[tr], y.iloc[tr])
            rows.append({"model": name, "fold": fold, "mae": mae(y.iloc[va], m.predict(X.iloc[va]))})
    cv = pd.DataFrame(rows).groupby("model")["mae"].agg(["mean", "std"]).sort_values("mean")
    return cv


# ------------------------------------------------------------------------------- train
def model_path(lh: Lakehouse, version: str) -> str:
    return f"{lh.model_dir()}/price_model_{version}.joblib"


def train_model(features: pd.DataFrame, train_cities: list[str], version: str,
                relative: bool = True) -> dict:
    df = prepare(features[features["city"].isin(train_cities)])
    train = df[(df["split"] == "train") & (df["price_status"] == "ok")].reset_index(drop=True)
    print(f"[train] {len(train):,} training rows from {train_cities} "
          f"(target: {'relative' if relative else 'absolute'} price)")
    cv = cross_validate(train, relative=relative)
    print("[train] 5-fold CV MAE on the training split (EUR):\n" + cv.round(2).to_string())
    best = next(m for m in cv.index if m != "baseline")
    if relative:
        feats = FEATURES_REL + ["city"]          # city is only used to look up the index
        idx = city_index(df)
        model = RelativePriceModel(candidates(CATEGORICAL_REL)[best]).fit(train[feats], train["price"], idx)
    else:
        feats, idx = FEATURES, None
        model = candidates()[best].fit(train[feats], train["price"])
    baseline = MedianBaseline().fit(train, train["price"])
    return {
        "version": version, "model_name": best, "model": model, "baseline": baseline,
        "target": "relative" if relative else "log_price", "city_index": idx,
        "features": feats, "train_cities": list(train_cities), "n_train": len(train),
        "train_nbhd_keys": sorted(train["nbhd_key"].unique()),
        "cv": cv.reset_index().to_dict(orient="records"),
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "price_rule": f"{config.PRICE_LOW} <= price <= {config.PRICE_HIGH_MULTIPLE} x city median",
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


def run_train(lh: Lakehouse, train_cities: list[str], version: str, relative: bool = True) -> dict:
    feats = (lh.read("gold_listing_features").where(F.col("city").isin(train_cities)).toPandas())
    bundle = train_model(feats, train_cities, version, relative=relative)
    path = save_bundle(lh, bundle)
    log_to_mlflow(bundle, path)
    print(f"[train] saved {bundle['model_name']} as {path}")
    return {"version": version, "model": bundle["model_name"], "path": path, "n_train": bundle["n_train"]}


# ------------------------------------------------------------------------------- score
def seen_neighbourhoods(bundle: dict) -> set:
    if bundle.get("train_nbhd_keys"):
        return set(bundle["train_nbhd_keys"])
    try:   # v1 bundles: read the categories of the fitted TargetEncoder
        te = bundle["model"].regressor_.named_steps["pre"].named_transformers_["te"]
        return set(te.categories_[0])
    except Exception:
        return set()


def predict_frame(bundle: dict, features: pd.DataFrame) -> pd.DataFrame:
    df = prepare(features)
    if bundle.get("target") == "relative":
        pred = bundle["model"].predict(df[bundle["features"]], index=city_index(df))
    else:
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


def coverage(lh: Lakehouse, bundle: dict, feats: pd.DataFrame, city: str) -> dict:
    """[city-C pipeline fix P3] week 1 scored an unseen city without a single warning.  Now
    every scoring run records how much of the input the model has actually seen."""
    from .silver import dq_frame
    df = prepare(feats)
    seen = seen_neighbourhoods(bundle)
    ok = df[df["price_status"] == "ok"]
    idx = bundle.get("city_index") or {}
    # a city the model was trained on is compared with its own level, a new city with the pooled one
    train_level = (idx[city] if city in idx
                   else float(np.median(list(idx.values()))) if idx else bundle["baseline"].global_)
    ratio = round(float(ok["price"].median() / train_level), 3) if len(ok) else -1.0
    cov = {
        "city_in_training_cities": float(city in bundle["train_cities"]),
        "unseen_neighbourhood_share": round(float((~df["nbhd_key"].isin(seen)).mean()), 4) if seen else -1.0,
        "median_price_vs_training_level": ratio,
    }
    checks = {}
    if not cov["city_in_training_cities"]:
        checks["city_in_training_cities"] = "warn"
        print(f"[score] WARNING {city} is not a training city of model {bundle['version']}")
    if cov["unseen_neighbourhood_share"] > config.UNSEEN_NEIGHBOURHOOD_WARN:
        checks["unseen_neighbourhood_share"] = "warn"
        print(f"[score] WARNING {cov['unseen_neighbourhood_share']:.0%} of listings are in "
              f"neighbourhoods the model never saw")
    if ratio > 1.5 or 0 < ratio < 0.67:
        checks["median_price_vs_training_level"] = "warn"
        print(f"[score] WARNING median price is {ratio}x the training price level")
    layer = f"score_{bundle['version']}"
    lh.write_city(dq_frame(lh, city, layer, cov, checks), "dq_metrics", city,
                  merge_schema=True, extra={"layer": layer})
    return cov


def run_score(lh: Lakehouse, city: str, version: str) -> dict:
    bundle = load_bundle(lh, version)
    feats = lh.read("gold_listing_features").where(F.col("city") == city).toPandas()
    if feats.empty:
        raise ValueError(f"[score] no gold features for {city}")
    preds = predict_frame(bundle, feats)
    coverage(lh, bundle, feats, city)
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
        if role == "training_city":
            groups.append((ver, city, role, "test split", g[g["split"] == "test"]))
        else:   # new city: all listings, plus its test split for comparison with retrained models
            groups.append((ver, city, role, "all listings", g))
            groups.append((ver, city, role, "test split", g[g["split"] == "test"]))
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
