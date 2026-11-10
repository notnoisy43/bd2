# Genie space: "Benelux Airbnb analyst"

Create it in the workspace: **Genie → New → choose the SQL warehouse "Serverless Starter Warehouse"**, then paste the
sections below. First run `notebooks/07_uc_comments.sql` once: Genie reads Unity Catalog table and column comments,
and that does more for answer quality than anything else.

## 1. Tables (Data → Add)

Five gold tables only. Fewer, well-described tables mean fewer wrong joins:

| table | use it for |
|---|---|
| `workspace.airbnb_benelux.gold_city_metrics` | one row per city: listings, hosts, median/mean price, availability, rating, shares |
| `workspace.airbnb_benelux.gold_room_type_metrics` | one row per city x room type: listings, share, median price |
| `workspace.airbnb_benelux.gold_neighbourhood_metrics` | one row per city x neighbourhood: listings, median price, rating |
| `workspace.airbnb_benelux.gold_predictions` | one row per listing x model version: actual vs predicted price, error |
| `workspace.airbnb_benelux.gold_model_metrics` | one row per model version x city: MAE of model vs baseline |

`silver_listings` is deliberately **not** added: it has 70 columns, and Genie would compute medians over outlier and missing prices.

## 2. General instructions (Instructions → Text)

```
You answer questions about Airbnb listings in the Benelux region for a short-term-rental analytics company.

Cities
- "City A" means Brussels, "City B" means Antwerp, "City C" (the new client) means Amsterdam.
- The column city holds lowercase slugs: 'brussels', 'antwerp', 'amsterdam'. Always filter with lowercase.

Prices
- All prices are per night in EUR. Never convert currencies.
- "Median price" questions: use the precomputed median_price columns of gold_city_metrics,
  gold_room_type_metrics or gold_neighbourhood_metrics. They exclude missing prices and outliers
  (below 15 EUR or above 1000 EUR). Do not recompute medians from gold_predictions unless the question is about the model.
- "Entire home(s)" / "whole apartment" = room_type 'Entire home/apt'. "Private room" = 'Private room'.

Model
- gold_predictions.error = predicted_price - price. error < 0 means the model UNDER-prices the listing
  (predicts less than the real price); error > 0 means it OVER-prices it.
- Evaluate the model only on honest rows: price_status = 'ok' AND (split = 'test' OR city_role = 'new_city').
  Training-split rows of Brussels/Antwerp were seen during training and must not be used for error statistics.
- If several model versions exist and the question does not say which, use the latest model_version per city and say which one you used.
- MAE = mean absolute error = AVG(abs_error). The baseline is the median price for the same room type and number of guests (baseline_price).
- For "where" questions about the model, group by city and neighbourhood and require at least 15 listings.

Style
- Round euros to whole numbers, percentages to one decimal. Mention the city and, for model questions, the model version.
```

## 3. Example SQL queries (Instructions → SQL queries; Genie uses these as trusted patterns)

**What is the median price of entire homes in City A?**  Expected: €138 (Brussels, 3,509 entire homes)
```sql
SELECT city, room_type, listings, median_price
FROM workspace.airbnb_benelux.gold_room_type_metrics
WHERE city = 'brussels' AND room_type = 'Entire home/apt';
```

**Which neighbourhoods in City B have the most listings?**  Expected: Historisch Centrum (397), Amandus - Atheneum (219), Theaterbuurt-Meir (194), ...
```sql
SELECT neighbourhood, listings, median_price
FROM workspace.airbnb_benelux.gold_neighbourhood_metrics
WHERE city = 'antwerp'
ORDER BY listings DESC
LIMIT 10;
```

**Where does the model under-price listings the most?**  Expected (v1, test split): Saint-Gilles, Brussels (mean error −€12, 79 listings), then Forest and Bruxelles
```sql
SELECT model_version, city, neighbourhood, COUNT(*) AS listings,
       ROUND(AVG(error), 1) AS mean_error_eur, ROUND(AVG(abs_error), 1) AS mae_eur
FROM workspace.airbnb_benelux.gold_predictions
WHERE price_status = 'ok' AND (split = 'test' OR city_role = 'new_city')
GROUP BY model_version, city, neighbourhood
HAVING COUNT(*) >= 15
ORDER BY mean_error_eur ASC
LIMIT 10;
```

**How much better is the model than the baseline?**
```sql
SELECT model_version, city, evaluation_set, n, mae_baseline, mae_model,
       ROUND(improvement_vs_baseline * 100, 1) AS improvement_pct
FROM workspace.airbnb_benelux.gold_model_metrics
ORDER BY model_version, city;
```

**Compare the cities**
```sql
SELECT city, listings, hosts, median_price, avg_availability_365, avg_review_rating,
       share_entire_home, share_multi_listing_host
FROM workspace.airbnb_benelux.gold_city_metrics
ORDER BY listings DESC;
```

## 4. Sample questions shown to users (Settings → Sample questions)

* What is the median price of entire homes in City A?
* Which neighbourhoods in City B have the most listings?
* Where does the model under-price listings the most?
* How much better than the baseline is the model in each city?
* Which city has the highest share of multi-listing hosts?
* Show the 10 listings in Antwerp the model over-prices most.

## 5. Configuration tricks and why we used them

*(Check them against the live space during the demo; the expected answers in section 3 come from the local run of the same pipeline.)*

1. **Pre-aggregated gold tables** for medians. Genie writes `AVG` when asked for "typical" price and has no reliable `MEDIAN`
   habit, so the medians are precomputed with the same outlier rule as the dashboard. Answers then match the dashboard exactly.
2. **Domain mapping in the instructions**: City A/B/C → slugs, "entire home" → `'Entire home/apt'`, and the sign convention of `error`.
   "Under-price" is ambiguous (is it `error < 0` or `> 0`?), and the instruction pins it down.
3. **The honest-evaluation filter written out** (`split = 'test' OR city_role = 'new_city'`). Otherwise Genie happily averages
   errors over training rows, which reports a too-good MAE.
4. **UC comments on every column** (`07_uc_comments.sql`). Genie shows them in its reasoning and picks the right table.
5. **Example SQL for each sample question**, so the three assignment questions match a trusted query.
6. Leaving out `silver_listings` and `gold_listing_features`: wide tables with model-feature columns (`dist_centre_km`, amenity flags)
   give Genie more ways to pick the wrong column for a business question.
