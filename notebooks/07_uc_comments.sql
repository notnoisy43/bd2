-- Databricks notebook source
-- MAGIC %md
-- MAGIC # 07 Unity Catalog comments for Genie and the dashboard
-- MAGIC Run once after the first `airbnb_lakehouse` run. Genie reads these comments when it chooses tables and columns.

-- COMMAND ----------

USE CATALOG workspace;
USE SCHEMA airbnb_benelux;

-- COMMAND ----------

COMMENT ON TABLE gold_city_metrics IS 'One row per city (Inside Airbnb snapshot). Supply, price, availability, review and host metrics. Prices per night in EUR; price metrics exclude missing prices and outliers (<15 or >1000 EUR). City A = brussels, City B = antwerp, City C = amsterdam.';
ALTER TABLE gold_city_metrics ALTER COLUMN city COMMENT 'Lowercase city slug: brussels (City A), antwerp (City B), amsterdam (City C).';
ALTER TABLE gold_city_metrics ALTER COLUMN listings COMMENT 'Number of listings in the snapshot (all, including without price).';
ALTER TABLE gold_city_metrics ALTER COLUMN active_listings COMMENT 'Listings with availability in the next 365 days or a review in the last 12 months.';
ALTER TABLE gold_city_metrics ALTER COLUMN priced_listings COMMENT 'Listings with a valid price (price_status = ok).';
ALTER TABLE gold_city_metrics ALTER COLUMN hosts COMMENT 'Distinct hosts.';
ALTER TABLE gold_city_metrics ALTER COLUMN median_price COMMENT 'Median nightly price in EUR over listings with valid price.';
ALTER TABLE gold_city_metrics ALTER COLUMN avg_availability_365 COMMENT 'Average number of available nights in the next 365 days.';
ALTER TABLE gold_city_metrics ALTER COLUMN avg_review_rating COMMENT 'Average overall review score (0-5) over listings with reviews.';
ALTER TABLE gold_city_metrics ALTER COLUMN share_entire_home COMMENT 'Fraction (0-1) of listings that are Entire home/apt.';
ALTER TABLE gold_city_metrics ALTER COLUMN share_multi_listing_host COMMENT 'Fraction (0-1) of listings whose host has more than one listing in the city.';
ALTER TABLE gold_city_metrics ALTER COLUMN median_est_revenue_l365d COMMENT 'Median Inside Airbnb revenue estimate (price x estimated occupancy) for the last 12 months, EUR.';

-- COMMAND ----------

COMMENT ON TABLE gold_room_type_metrics IS 'One row per city and room type: number of listings, share of the city, median nightly price (EUR, valid prices only).';
ALTER TABLE gold_room_type_metrics ALTER COLUMN room_type COMMENT 'Entire home/apt, Private room, Hotel room or Shared room. "Entire home" means Entire home/apt.';
ALTER TABLE gold_room_type_metrics ALTER COLUMN median_price COMMENT 'Median nightly price in EUR for this city and room type (valid prices only).';
ALTER TABLE gold_room_type_metrics ALTER COLUMN share_of_city COMMENT 'Fraction (0-1) of the city''s listings with this room type.';

-- COMMAND ----------

COMMENT ON TABLE gold_neighbourhood_metrics IS 'One row per city and neighbourhood (Inside Airbnb neighbourhood_cleansed): listings, median nightly price EUR, average rating and availability.';
ALTER TABLE gold_neighbourhood_metrics ALTER COLUMN neighbourhood COMMENT 'Neighbourhood name as published by Inside Airbnb (communes in Brussels, districts in Antwerp).';
ALTER TABLE gold_neighbourhood_metrics ALTER COLUMN listings COMMENT 'Number of listings in the neighbourhood.';
ALTER TABLE gold_neighbourhood_metrics ALTER COLUMN median_price COMMENT 'Median nightly price EUR (valid prices only).';

-- COMMAND ----------

COMMENT ON TABLE gold_predictions IS 'One row per listing and model version: actual price, model prediction, baseline prediction and errors. For error statistics use only price_status = ''ok'' AND (split = ''test'' OR city_role = ''new_city'').';
ALTER TABLE gold_predictions ALTER COLUMN model_version COMMENT 'Model version, e.g. v1 (week-1 model trained on Brussels + Antwerp).';
ALTER TABLE gold_predictions ALTER COLUMN city_role COMMENT 'training_city if the model was trained on this city, new_city if the city was never seen by the model.';
ALTER TABLE gold_predictions ALTER COLUMN split COMMENT 'train or test (deterministic 80/20 split per listing). For training cities only test rows are honest.';
ALTER TABLE gold_predictions ALTER COLUMN price COMMENT 'Actual advertised nightly price EUR (NULL if the listing had no price).';
ALTER TABLE gold_predictions ALTER COLUMN price_status COMMENT 'ok, missing, low_outlier (<15 EUR) or high_outlier (>1000 EUR).';
ALTER TABLE gold_predictions ALTER COLUMN predicted_price COMMENT 'Model suggested nightly price EUR.';
ALTER TABLE gold_predictions ALTER COLUMN baseline_price COMMENT 'Baseline: training median price for the same room type and number of guests.';
ALTER TABLE gold_predictions ALTER COLUMN error COMMENT 'predicted_price - price. Negative = model UNDER-prices the listing; positive = OVER-prices.';
ALTER TABLE gold_predictions ALTER COLUMN abs_error COMMENT 'Absolute error of the model in EUR; MAE = AVG(abs_error).';
ALTER TABLE gold_predictions ALTER COLUMN baseline_abs_error COMMENT 'Absolute error of the baseline in EUR.';

-- COMMAND ----------

COMMENT ON TABLE gold_model_metrics IS 'Model quality per model version and city: MAE of the model and of the baseline in EUR on the honest evaluation set (test split for training cities, all priced listings for a new city). City ALL TRAINING CITIES pools the test splits of Brussels and Antwerp.';
ALTER TABLE gold_model_metrics ALTER COLUMN mae_model COMMENT 'Mean absolute error of the model, EUR.';
ALTER TABLE gold_model_metrics ALTER COLUMN mae_baseline COMMENT 'Mean absolute error of the baseline, EUR.';
ALTER TABLE gold_model_metrics ALTER COLUMN improvement_vs_baseline COMMENT 'Fraction 1 - mae_model / mae_baseline.';

-- COMMAND ----------

ALTER TABLE gold_predictions ALTER COLUMN model_version COMMENT 'v1 = week-1 model (Brussels+Antwerp, absolute price); v2 = relative-price model trained on Brussels+Antwerp; v3 = relative-price model retrained with Amsterdam. Production: v1 for brussels/antwerp, v3 for amsterdam.';
COMMENT ON TABLE dq_metrics IS 'Data-quality and scoring-coverage metrics per city and layer (silver, score_<version>). check_status = warn marks a value outside the expectation derived from the Brussels/Antwerp EDA.';
