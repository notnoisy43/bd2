-- ds_city
SELECT * FROM workspace.airbnb_benelux.gold_city_metrics;

-- ds_room
SELECT * FROM workspace.airbnb_benelux.gold_room_type_metrics;

-- ds_nbhd
SELECT city, neighbourhood, listings, priced_listings, median_price, avg_review_rating,
       avg_availability_365, concat(city, ' · ', neighbourhood) AS city_neighbourhood
FROM workspace.airbnb_benelux.gold_neighbourhood_metrics
WHERE priced_listings >= 10;

-- ds_compare
SELECT city, metric, value FROM workspace.airbnb_benelux.gold_city_metrics
LATERAL VIEW stack(8,
  'Median price (EUR)', median_price,
  'Mean price (EUR)', mean_price,
  'Avg days available / year', avg_availability_365,
  '% entire homes', share_entire_home * 100,
  '% multi-listing hosts', share_multi_listing_host * 100,
  '% superhosts', share_superhost * 100,
  '% long-stay (min 28 nights)', share_long_stay * 100,
  'Avg review rating x 20', avg_review_rating * 20) AS metric, value;

-- ds_price_hist
SELECT city, LEAST(FLOOR(price / 25) * 25, 500) AS price_bucket_eur,
       COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY city) AS share_of_listings
FROM workspace.airbnb_benelux.silver_listings
WHERE price_status = 'ok'
GROUP BY city, LEAST(FLOOR(price / 25) * 25, 500);

-- ds_metrics
SELECT model_version, city, city_role, evaluation_set, n, mae_baseline, mae_model,
       ROUND(improvement_vs_baseline * 100, 1) AS improvement_pct, median_abs_error_model,
       ROUND(mape_model * 100, 1) AS mape_pct, mean_error_model, median_price
FROM workspace.airbnb_benelux.gold_model_metrics
ORDER BY model_version, city_role DESC, city;

-- ds_eval
SELECT model_version, city, city_role, listing_id, name, room_type, neighbourhood, accommodates,
       price, predicted_price, baseline_price, error, abs_error, baseline_abs_error, listing_url
FROM workspace.airbnb_benelux.gold_predictions
WHERE price_status = 'ok' AND (city_role = 'new_city' OR split = 'test');

-- ds_err_room
SELECT model_version, city, room_type, COUNT(*) AS n,
       ROUND(AVG(abs_error), 1) AS mae_model, ROUND(AVG(baseline_abs_error), 1) AS mae_baseline
FROM workspace.airbnb_benelux.gold_predictions
WHERE price_status = 'ok' AND (city_role = 'new_city' OR split = 'test')
GROUP BY model_version, city, room_type
HAVING COUNT(*) >= 5;

-- ds_err_nbhd
SELECT model_version, city, neighbourhood, COUNT(*) AS n,
       ROUND(AVG(error), 1) AS mean_error, ROUND(AVG(abs_error), 1) AS mae_model,
       concat(city, ' · ', neighbourhood) AS city_neighbourhood
FROM workspace.airbnb_benelux.gold_predictions
WHERE price_status = 'ok' AND (city_role = 'new_city' OR split = 'test')
GROUP BY model_version, city, neighbourhood
HAVING COUNT(*) >= 15;

-- ds_extremes
SELECT * FROM (
  SELECT model_version, city, CASE WHEN error < 0 THEN 'under-priced' ELSE 'over-priced' END AS direction,
         name, room_type, neighbourhood, accommodates, price, predicted_price, error, listing_url,
         ROW_NUMBER() OVER (PARTITION BY model_version, city, error < 0 ORDER BY abs_error DESC) AS rnk
  FROM workspace.airbnb_benelux.gold_predictions
  WHERE price_status = 'ok' AND (city_role = 'new_city' OR split = 'test'))
WHERE rnk <= 10
ORDER BY model_version, city, direction, rnk;

