"""Generates dashboards/airbnb_benelux.lvdash.json (Databricks AI/BI dashboard) and
dashboards/queries.sql (the same dataset queries, readable / reusable).

--validate  runs every dataset query against the local lakehouse with Spark SQL.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT = "workspace.airbnb_benelux"

# Titles state findings (numbers from gold_* tables; refreshed after every run).
TITLES = {
    "overview": ("Brussels has twice Antwerp's listings, but prices are almost the same: median €125 vs €130",
                 "Per-city supply, price, availability, reviews and hosts. Price figures exclude missing and outlier prices (`price_status = 'ok'`)."),
    "compare": ("Antwerp is more 'professional' (84 % entire homes, 61 % multi-listing hosts); Brussels earns more per listing",
                "Side-by-side metrics for every city. The difference that matters for pricing is the mix, not the price level."),
    "model": ("The model beats the baseline by ~30 % in Brussels and Antwerp (MAE €35 vs €50), and under-prices the expensive tail",
              "Honest evaluation: 20 % test split for training cities, every priced listing for a new city."),
}

DATASETS = {
    "ds_city": f"SELECT * FROM {CAT}.gold_city_metrics",
    "ds_room": f"SELECT * FROM {CAT}.gold_room_type_metrics",
    "ds_nbhd": f"""SELECT city, neighbourhood, listings, priced_listings, median_price, avg_review_rating,
       avg_availability_365, concat(city, ' · ', neighbourhood) AS city_neighbourhood
FROM {CAT}.gold_neighbourhood_metrics
WHERE priced_listings >= 10""",
    "ds_compare": f"""SELECT city, metric, value FROM {CAT}.gold_city_metrics
LATERAL VIEW stack(8,
  'Median price (EUR)', median_price,
  'Mean price (EUR)', mean_price,
  'Avg days available / year', avg_availability_365,
  '% entire homes', share_entire_home * 100,
  '% multi-listing hosts', share_multi_listing_host * 100,
  '% superhosts', share_superhost * 100,
  '% long-stay (min 28 nights)', share_long_stay * 100,
  'Avg review rating x 20', avg_review_rating * 20) AS metric, value""",
    "ds_price_hist": f"""SELECT city, LEAST(FLOOR(price / 25) * 25, 500) AS price_bucket_eur,
       COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY city) AS share_of_listings
FROM {CAT}.silver_listings
WHERE price_status = 'ok'
GROUP BY city, LEAST(FLOOR(price / 25) * 25, 500)""",
    "ds_metrics": f"""SELECT model_version, city, city_role, evaluation_set, n, mae_baseline, mae_model,
       ROUND(improvement_vs_baseline * 100, 1) AS improvement_pct, median_abs_error_model,
       ROUND(mape_model * 100, 1) AS mape_pct, mean_error_model, median_price
FROM {CAT}.gold_model_metrics
ORDER BY model_version, city_role DESC, city""",
    "ds_eval": f"""SELECT model_version, city, city_role, listing_id, name, room_type, neighbourhood, accommodates,
       price, predicted_price, baseline_price, error, abs_error, baseline_abs_error, listing_url
FROM {CAT}.gold_predictions
WHERE price_status = 'ok' AND (city_role = 'new_city' OR split = 'test')""",
    "ds_err_room": f"""SELECT model_version, city, room_type, COUNT(*) AS n,
       ROUND(AVG(abs_error), 1) AS mae_model, ROUND(AVG(baseline_abs_error), 1) AS mae_baseline
FROM {CAT}.gold_predictions
WHERE price_status = 'ok' AND (city_role = 'new_city' OR split = 'test')
GROUP BY model_version, city, room_type
HAVING COUNT(*) >= 5""",
    "ds_err_nbhd": f"""SELECT model_version, city, neighbourhood, COUNT(*) AS n,
       ROUND(AVG(error), 1) AS mean_error, ROUND(AVG(abs_error), 1) AS mae_model,
       concat(city, ' · ', neighbourhood) AS city_neighbourhood
FROM {CAT}.gold_predictions
WHERE price_status = 'ok' AND (city_role = 'new_city' OR split = 'test')
GROUP BY model_version, city, neighbourhood
HAVING COUNT(*) >= 15""",
    "ds_extremes": f"""SELECT * FROM (
  SELECT model_version, city, CASE WHEN error < 0 THEN 'under-priced' ELSE 'over-priced' END AS direction,
         name, room_type, neighbourhood, accommodates, price, predicted_price, error, listing_url,
         ROW_NUMBER() OVER (PARTITION BY model_version, city, error < 0 ORDER BY abs_error DESC) AS rnk
  FROM {CAT}.gold_predictions
  WHERE price_status = 'ok' AND (city_role = 'new_city' OR split = 'test'))
WHERE rnk <= 10
ORDER BY model_version, city, direction, rnk""",
}

_n = [0]


def _name(prefix):
    _n[0] += 1
    return f"{prefix}_{_n[0]}"


def f(name, expr=None):
    return {"name": name, "expression": expr or f"`{name}`"}


def query(ds, fields, agg=False):
    return [{"name": "main_query", "query": {"datasetName": ds, "fields": fields, "disaggregated": not agg}}]


def frame(title, desc=None):
    fr = {"showTitle": True, "title": title}
    if desc:
        fr.update({"showDescription": True, "description": desc})
    return fr


def text(lines, x, y, w, h):
    return {"widget": {"name": _name("text"), "multilineTextboxSpec": {"lines": lines}},
            "position": {"x": x, "y": y, "width": w, "height": h}}


def counter(ds, field, expr, title, x, y, w=1, h=2):
    return {"widget": {"name": _name("counter"), "queries": query(ds, [f(field, expr)], agg=True),
                       "spec": {"version": 2, "widgetType": "counter",
                                "encodings": {"value": {"fieldName": field, "displayName": title}},
                                "frame": frame(title)}},
            "position": {"x": x, "y": y, "width": w, "height": h}}


def bar(ds, xf, yf, title, pos, color=None, horizontal=False, agg=None, desc=None, xlabel=None, ylabel=None):
    fields = [f(xf), f(yf, agg)] + ([f(color)] if color and color not in (xf,) else [])
    cat, val = ("y", "x") if horizontal else ("x", "y")
    enc = {cat: {"fieldName": xf, "scale": {"type": "categorical"}, "displayName": xlabel or xf},
           val: {"fieldName": yf, "scale": {"type": "quantitative"}, "displayName": ylabel or yf}}
    if horizontal:
        enc[cat]["scale"]["sort"] = {"by": "x-reversed"}
    if color:
        enc["color"] = {"fieldName": color, "scale": {"type": "categorical"}, "displayName": color}
    return {"widget": {"name": _name("bar"), "queries": query(ds, fields, agg=bool(agg)),
                       "spec": {"version": 3, "widgetType": "bar", "encodings": enc,
                                "frame": frame(title, desc)}},
            "position": dict(zip("xywh", pos)) | {}}


def scatter(ds, xf, yf, color, title, pos, desc=None):
    enc = {"x": {"fieldName": xf, "scale": {"type": "quantitative"}, "displayName": xf},
           "y": {"fieldName": yf, "scale": {"type": "quantitative"}, "displayName": yf},
           "color": {"fieldName": color, "scale": {"type": "categorical"}, "displayName": color}}
    return {"widget": {"name": _name("scatter"), "queries": query(ds, [f(xf), f(yf), f(color)]),
                       "spec": {"version": 3, "widgetType": "scatter", "encodings": enc,
                                "frame": frame(title, desc)}},
            "position": dict(zip("xywh", pos))}


def table(ds, cols, title, pos, desc=None):
    columns = [{"fieldName": c, "displayName": c, "title": c, "type": "string", "displayAs": "string",
                "visible": True, "order": i, "booleanValues": ["false", "true"], "alignContent": "left"}
               for i, c in enumerate(cols)]
    return {"widget": {"name": _name("table"), "queries": query(ds, [f(c) for c in cols]),
                       "spec": {"version": 1, "widgetType": "table",
                                "encodings": {"columns": columns},
                                "invisibleColumns": [], "allowHTMLByDefault": False,
                                "itemsPerPage": 25, "paginationSize": "default",
                                "condensed": True, "withRowNumber": False,
                                "frame": frame(title, desc)}},
            "position": dict(zip("xywh", pos))}


def fix_pos(item):
    p = item["position"]
    item["position"] = {"x": p["x"], "y": p["y"], "width": p["w"], "height": p["h"]} if "w" in p else p
    return item


def page(name, display, items):
    return {"name": name, "displayName": display, "pageType": "PAGE_TYPE_CANVAS",
            "layout": [fix_pos(i) for i in items]}


def build():
    t = TITLES
    overview = page("overview", "City overview", [
        text([f"## {t['overview'][0]}", "", t["overview"][1]], 0, 0, 6, 2),
        table("ds_city", ["city", "snapshot_date", "listings", "active_listings", "priced_listings", "hosts",
                          "median_price", "p25_price", "p75_price", "avg_availability_365",
                          "avg_review_rating", "share_entire_home", "share_multi_listing_host"],
              "Per-city summary (EUR)", (0, 2, 6, 4)),
        bar("ds_city", "city", "listings", "Listings", (0, 6, 2, 5), color="city"),
        bar("ds_city", "city", "median_price", "Median price per night (EUR)", (2, 6, 2, 5), color="city"),
        bar("ds_city", "city", "avg_availability_365", "Average days available in the next year", (4, 6, 2, 5), color="city"),
        bar("ds_room", "room_type", "listings", "Room-type mix", (0, 11, 3, 6), color="city",
            desc="Entire homes dominate both cities; private rooms are more common in Brussels."),
        bar("ds_city", "city", "avg_review_rating", "Average review rating (0-5)", (3, 11, 3, 6), color="city",
            desc="Ratings are uniformly high (4.7 / 5): they barely separate listings."),
        bar("ds_nbhd", "city_neighbourhood", "median_price", "Neighbourhoods by median price (>= 10 priced listings)",
            (0, 17, 6, 10), color="city", horizontal=True, xlabel="neighbourhood", ylabel="median price (EUR)"),
    ])
    compare = page("compare", "Cross-city comparison", [
        text([f"## {t['compare'][0]}", "", t["compare"][1]], 0, 0, 6, 2),
        bar("ds_compare", "metric", "value", "Key metrics side by side", (0, 2, 6, 7), color="city",
            horizontal=True, desc="Shares in %, rating scaled x20 to share the axis."),
        bar("ds_price_hist", "price_bucket_eur", "share_of_listings", "Price distribution (EUR buckets of 25, 500 = 500+)",
            (0, 9, 6, 6), color="city"),
        bar("ds_room", "room_type", "median_price", "Median price by room type (EUR)", (0, 15, 3, 6), color="city"),
        bar("ds_city", "city", "median_est_revenue_l365d", "Median estimated revenue last 12 months (EUR)", (3, 15, 3, 6), color="city",
            desc="Inside Airbnb estimate = price x estimated occupancy (dashboard only, never a model feature)."),
    ])
    model = page("model", "Model", [
        text([f"## {t['model'][0]}", "", t["model"][1]], 0, 0, 6, 2),
        table("ds_metrics", ["model_version", "city", "city_role", "evaluation_set", "n", "mae_baseline", "mae_model",
                             "improvement_pct", "median_abs_error_model", "mape_pct", "mean_error_model", "median_price"],
              "MAE (EUR): model vs baseline", (0, 2, 6, 4)),
        scatter("ds_eval", "price", "predicted_price", "city", "Predicted vs actual price (EUR)", (0, 6, 3, 8),
                desc="Points below the diagonal are under-priced by the model."),
        bar("ds_err_room", "room_type", "mae_model", "Model MAE by room type (EUR)", (3, 6, 3, 8), color="city"),
        bar("ds_err_nbhd", "city_neighbourhood", "mean_error", "Mean error by neighbourhood (EUR, < 0 = under-priced)",
            (0, 14, 6, 10), color="city", horizontal=True, xlabel="neighbourhood", ylabel="mean error (EUR)"),
        table("ds_extremes", ["model_version", "city", "direction", "rnk", "name", "room_type", "neighbourhood",
                              "accommodates", "price", "predicted_price", "error", "listing_url"],
              "Listings the model under- / over-prices most (top 10 each)", (0, 24, 6, 9)),
    ])
    return {"datasets": [{"name": k, "displayName": k.replace("ds_", ""), "queryLines": [v]} for k, v in DATASETS.items()],
            "pages": [overview, compare, model]}


def validate():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    from peek import rd
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master("local[2]").config("spark.ui.enabled", "false")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    for tname in ["gold_city_metrics", "gold_room_type_metrics", "gold_neighbourhood_metrics",
                  "gold_model_metrics", "gold_predictions", "silver_listings"]:
        pdf = rd(tname)
        if "amenities" in pdf:
            pdf = pdf.drop(columns=["amenities"])
        spark.createDataFrame(pdf).createOrReplaceTempView(tname)
    for k, sql in DATASETS.items():
        df = spark.sql(sql.replace(CAT + ".", ""))
        print(f"{k}: {df.count()} rows, columns={df.columns}")


if __name__ == "__main__":
    out = build()
    os.makedirs(os.path.join(ROOT, "dashboards"), exist_ok=True)
    with open(os.path.join(ROOT, "dashboards", "airbnb_benelux.lvdash.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(ROOT, "dashboards", "queries.sql"), "w", encoding="utf-8") as fh:
        for k, v in DATASETS.items():
            fh.write(f"-- {k}\n{v};\n\n")
    print("dashboard written")
    if "--validate" in sys.argv:
        validate()
