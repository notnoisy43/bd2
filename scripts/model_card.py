"""Writes docs/model_<version>.md: CV table, test metrics, permutation importance, error by segment.
Every number except the CV table comes from the test split."""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from airbnb_lakehouse.model import prepare  # noqa: E402
from peek import rd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--version", default="v1")
a = ap.parse_args()

bundle = joblib.load(os.path.join(ROOT, "data", "models", f"price_model_{a.version}.joblib"))
feats = prepare(rd("gold_listing_features"))
feats = feats[feats.city.isin(bundle["train_cities"])]
test = feats[(feats.split == "test") & (feats.price_status == "ok")]
metrics = rd("gold_model_metrics")
metrics = metrics[(metrics.model_version == a.version) & (metrics.city_role == "training_city")]
preds = rd("gold_predictions")
preds = preds[(preds.model_version == a.version) & preds.city.isin(bundle["train_cities"])
              & (preds.split == "test") & (preds.price_status == "ok")]

mae = lambda y, p: float(np.mean(np.abs(y - p)))  # noqa: E731
relative = bundle.get("target") == "relative"
pi = permutation_importance(bundle["model"], test[bundle["features"]], test["price"],
                            scoring=lambda est, X, y: -mae(y, est.predict(X)),
                            n_repeats=5, random_state=0)
imp = (pd.DataFrame({"feature": bundle["features"], "MAE increase (EUR)": pi.importances_mean})
       .query("not (feature == 'city' and @relative)")
       .sort_values("MAE increase (EUR)", ascending=False).round(2))

by_rt = (preds.groupby(["city", "room_type"])
         .agg(n=("price", "size"), median_price=("price", "median"), mae_model=("abs_error", "mean"),
              mae_baseline=("baseline_abs_error", "mean"), mean_error=("error", "mean")).round(1))
by_band = (preds.assign(price_band=pd.cut(preds.price, [0, 75, 125, 200, 400, 1000]))
           .groupby("price_band", observed=True)
           .agg(n=("price", "size"), mae_model=("abs_error", "mean"), mean_error=("error", "mean")).round(1))

cv = pd.DataFrame(bundle["cv"]).round(2)
md = f"""# Price model {a.version}

* **Model**: `{bundle['model_name']}` (scikit-learn HistGradientBoostingRegressor on {'log(price / city price index)' if relative else 'log(price)'}, exp back), chosen by
  5-fold CV on the training split; the test split was not used for any choice.
* **Trained on**: {', '.join(bundle['train_cities'])}: {bundle['n_train']:,} listings (`split='train'`, `price_status='ok'`), trained at {bundle['trained_at'][:19]} UTC.
* **Baseline**: median training price for the same (room_type, accommodates), falling back to the room-type median and then the global median.
* **Price rule**: {bundle.get('price_rule') or f"{bundle['price_bounds'][0]:.0f} <= price <= {bundle['price_bounds'][1]:.0f} EUR"} (training and primary evaluation).
* **City price index** (median training-split price): {bundle.get('city_index') or 'n/a (absolute model)'}

## Model choice: 5-fold CV MAE on the training split (EUR)

{cv.to_markdown(index=False)}

## Test-set error (EUR), the numbers we report

{metrics[['city','evaluation_set','n','mae_model','mae_baseline','improvement_vs_baseline','median_abs_error_model','mape_model','mean_error_model']].to_markdown(index=False)}

`mean_error_model` < 0 means that on average the model predicts below the actual price. Because the model fits log(price), its
predictions sit near the median rather than the mean, so expensive listings are under-predicted in euro terms.

## Error by room type (test split)

{by_rt.to_markdown()}

## Error by price band (test split)

{by_band.to_markdown()}

The error grows with price: listings above €400 are rare, and the features do not capture what makes them expensive
(view, design, exact location).

## Which features matter: permutation importance on the test split

How much the test MAE rises when one feature is shuffled (5 repeats):

{imp.to_markdown(index=False)}
"""
out = os.path.join(ROOT, "docs", f"model_{a.version}.md")
open(out, "w", encoding="utf-8").write(md)
print(md)
