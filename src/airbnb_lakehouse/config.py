"""Project-wide constants. Nothing in here is city specific: a city is a runtime parameter."""

REGION = "Benelux"
CURRENCY = "EUR"

# Inside Airbnb publishes the latest snapshot of every city on this page.  The pipeline
# resolves `city -> (snapshot_date, url)` from it, so adding a city needs no code change.
INDEX_URL = "https://insideairbnb.com/get-the-data/"
USER_AGENT = "Mozilla/5.0 (compatible; airbnb-lakehouse-student-project)"

DEFAULT_CATALOG = "workspace"          # Databricks Free Edition default catalog
DEFAULT_SCHEMA = "airbnb_benelux"
RAW_VOLUME = "raw"                     # /Volumes/<catalog>/<schema>/raw/<city>/<date>/listings.csv.gz
MODEL_VOLUME = "models"                # /Volumes/<catalog>/<schema>/models/price_model_<version>.joblib

TRAIN_CITIES = ["brussels", "antwerp"]

# ---- price outlier policy (see EDA notebook, section 4) ---------------------------------
# Prices are per night in EUR.  Below 15 EUR is not a real nightly price in Benelux;
# above 1000 EUR are mostly 0-review placeholder prices or whole-villa event rentals
# (0.4 % of priced listings in A+B).  Such rows stay in silver (flagged), but are excluded
# from model training and from the primary MAE.
PRICE_LOW = 15.0
PRICE_HIGH = 1000.0           # week-1 rule (kept for reference / v1 metadata)
# [city-C pipeline fix P1] the ceiling is relative to the city's median price:
# 8 x 125.5 = 1004 EUR in Brussels (= the week-1 rule), 8 x 285 = 2282 EUR in Amsterdam.
PRICE_HIGH_MULTIPLE = 8.0

# [city-C pipeline fix P2] data-quality expectations, ranges from the A/B EDA.  Violations are
# logged as warnings and stored in dq_metrics.check_status = 'warn' (they do not stop the job:
# a new city may legitimately differ, but somebody has to look).
DQ_EXPECTATIONS = {
    "silver_rows": (500, None),
    "optional_columns_missing": (None, 0),
    "price_missing_share": (None, 0.30),            # A/B: 0.16 / 0.20
    "availability_zero_share": (None, 0.25),        # A/B: 0.13 / 0.12
    "high_outlier_share_of_priced": (None, 0.02),   # A/B: 0.005
}
# [city-C pipeline fix P3] scoring coverage: warn when this share of scored rows has a
# neighbourhood the model never saw, or a city outside the model's training cities.
UNSEEN_NEIGHBOURHOOD_WARN = 0.2

# ---- train / test split ------------------------------------------------------------------
# Deterministic, listing-level split: xxhash64(id) mod 100 < 20 -> test.  It is computed in
# the gold layer, so every rerun (and every consumer) sees the same split.
TEST_PCT = 20
SPLIT_SEED = 42

# Long-stay listings: Airbnb quotes these with a monthly discount, so the advertised nightly
# price is a monthly-discounted price (seen in price_quote_raw.is_monthly_priced).
LONG_STAY_MIN_NIGHTS = 28
