# The new client: city C = Amsterdam

| | |
|---|---|
| City C | **Amsterdam**, snapshot **2026-09-16**, 10,612 listings |
| Frozen state used for the first run | git tag `week1-final` (commit `027a4c4`): pipeline code + model `price_model_v1.joblib` (sha256 `74c475d6…2a40`) |
| Logs | [`first_run_etl.log`](first_run_etl.log), [`first_run_score.log`](first_run_score.log), [`first_run_metrics.csv`](first_run_metrics.csv), [`after_pipeline_fixes.log`](after_pipeline_fixes.log), [`model_changes.log`](model_changes.log) |
| Every change since | `git diff week1-final` (each change is tagged in code with `[city-C pipeline fix Pn]` or `[city-C model change Mn]`) |

## 1. First run: unchanged pipeline, unchanged model

Before the run, `git diff week1-final -- src notebooks resources/jobs.yml scripts/run_local.py` was empty, and the model file hash
matched the tag. Commands: `run_local.py etl amsterdam`, `run_local.py score amsterdam --version v1`, `run_local.py report`.

**Did it run?** Yes, end to end, exit code 0, with no errors and no warnings.

* Bronze resolved `amsterdam` on the Inside Airbnb index by itself
  (`the-netherlands/north-holland/amsterdam/2026-09-16`), downloaded 5.9 MB, and loaded 10,612 rows x 90 columns.
* Same 90 columns as Brussels and Antwerp: the column contract found no missing required or optional column. Prices are `$`-formatted EUR, and
  every quote in `price_quote_raw` is in EUR. Parsing, `try_cast` and de-duplication changed nothing (0 rows dropped).

**What came out**

| | Brussels | Antwerp | **Amsterdam** |
|---|---:|---:|---:|
| listings | 4,850 | 2,463 | **10,612** |
| price missing | 15.5 % | 20.1 % | **43.1 %** |
| `availability_365 = 0` | 13.1 % | 12.3 % | **42.0 %** |
| flagged high outlier (> €1,000, week-1 rule) | 20 | 10 | **112** |
| median price (valid) | €125 | €130.5 | **€285** |
| median price per guest | €48 | €45 | **€108** |
| multi-listing hosts | 53 % | 61 % | **19 %** |
| long-stay (min. 28 nights) | 8.3 % | 6.8 % | **1.1 %** |

**MAE on city C (first run): €158.60** on all 5,927 priced listings (€156.03 on its 20 % test split), against €34.91 on the A+B test sets.
The baseline trained on A+B scored €195.54. The model beat a bad baseline, but its predictions were useless for this client:

* mean error **−€155.52**, and **95 %** of Amsterdam listings were under-priced;
* the median prediction was **0.55 x** the actual price, and the highest prediction for any listing was €601;
* entire homes had a median actual price of €327 and a median prediction of €165.

**The most important first-run finding is that nothing warned us.** A 43 % missing target, 2.2x higher prices and 100 % unknown
neighbourhoods all went through silently, and a confident-looking table of predictions came out.

## 2. Diagnosis

| symptom | evidence | cause |
|---|---|---|
| systematic under-pricing | one constant factor (x1.84, an *oracle* fitted on Amsterdam's own prices, diagnostic only) applied to v1's predictions cuts the MAE from €158.6 to €87.7 | **price level**: Amsterdam is 2.2x Brussels. v1 learned absolute EUR prices, and gradient-boosted trees cannot predict above the range they were trained on |
| `city` feature useless | importance 2.8 € in v1. For Amsterdam the one-hot is all zeros ("neither Brussels nor Antwerp") | a feature that has no meaning for a new city |
| neighbourhood feature useless | 100 % of Amsterdam's 22 neighbourhoods unseen, so `TargetEncoder` falls back to the global A+B mean | expected, but it removes the 2nd-best location signal |
| 112 "high outliers" | median 4 guests, 3 bedrooms, 53 % with 10 or more reviews: genuine canal houses and family homes, not placeholders | **pipeline**: the €1,000 ceiling was tuned to Belgian prices (8x the Brussels median) |
| 43 % missing price | 94 % of listings with `availability_365 = 0` have no price, the same mechanism as in A/B, but 42 % of listings have zero availability | most likely Amsterdam's 30-nights-per-year cap on renting out entire homes: many calendars are closed. The pipeline handled it correctly (flag and exclude), but should have *said* so |
| different price structure | entire home / private room price ratio 1.89 (Brussels 1.50, Antwerp 1.75). Per-guest price falls faster with group size. Price falls with distance from the centre more steeply (listings 5+ km out: 0.75x the city median in Amsterdam, 0.94-0.99x in Belgium) | room type, guests and distance mean something different in Amsterdam |
| host features shift meaning | 19 % multi-listing hosts vs 53-61 %. Regulation leaves mostly private individuals | `calculated_host_listings_count` separates different things in Amsterdam |

## 3. Every change, classified

| # | kind | change | reason |
|---|---|---|---|
| P1 | **pipeline fix** | high-outlier ceiling = **8 x the city's median price** instead of a fixed €1,000 (`silver.py`, `config.PRICE_HIGH_MULTIPLE`); `city_median_price` and `price_high_threshold` columns in silver | a cleaning rule must not encode one city's price level. It is identical for Brussels (8 x 125.5 = €1,004), moves 1 Antwerp training listing, and stops deleting 103 genuine Amsterdam homes |
| P2 | **pipeline fix** | data-quality expectations in silver (`config.DQ_EXPECTATIONS`): price-missing share, zero-availability share, outlier share, missing optional columns, row count. Results go to `dq_metrics.check_status` (`pass`/`warn`) and the job log | the first run accepted a 43 % missing target without a word. Now Amsterdam raises 2 warnings, and Brussels/Antwerp pass |
| P3 | **pipeline fix** | scoring coverage report (`model.coverage`): is the city a training city, what share of neighbourhoods is unseen, and the price level vs the training level. Goes to `dq_metrics` layer `score_<version>` with warnings | a model must say when it is extrapolating. For v1 on Amsterdam: not a training city / 100 % unseen neighbourhoods / 2.25x price level |
| P4 | pipeline fix (supporting) | `dq_metrics` written per (city, layer) with schema merge | P2/P3 add a column and a second writer to the same table; per-city overwrite would otherwise delete the other layer's rows |
| P5 | pipeline fix (found while writing the week-2 run instructions, not caused by Amsterdam) | job parameters `cities` / `train_cities` are now `+`-separated and turned into the for-each list by a `plan` task | the Databricks CLI splits `--params` on commas, so the week-1 JSON-list parameter could not be passed from the command line. Adding a city should be one parameter value |
| P6 | pipeline fix (found on the first Databricks deploy, not caused by Amsterdam) | in `resources/jobs.yml`, `score_each_city` now depends on `retrain_requested` with `outcome: "false"` | the Jobs API rejects a dependency on an if/else task without an outcome (`400 INVALID_PARAMETER_VALUE`), so `databricks bundle deploy` failed for the orchestrator job. The week-1 job definition had the same defect; the job YAML had never been checked against the real API |
| M1 | **model change** | **relative target**: learn `log(price / city price index)`, where the index is the median price of the city's priced *training-split* listings, and multiply back at prediction (`RelativePriceModel`). For a new city the index comes from that city's own training split: one aggregate number, no test labels | the price level the model never saw. The model now learns *structure* ("a 4-guest entire home 1 km from the centre costs 1.4x the city median") rather than Belgian euro amounts |
| M2 | **model change** | drop the one-hot `city` feature in relative models | it carries no information for an unseen city; the city now enters only through its index |
| M3 | **model change** | **retrain with Amsterdam's training split** (v3 = A + B + C train, same features, same CV procedure) | M1 fixes the level but not Amsterdam's different structure (room-type ratio, distance gradient). The snapshot contains Amsterdam prices, so a new client city comes with labels; using its 80 % training split is legitimate onboarding, and its 20 % test split stays untouched |

Not changed, because it worked on unseen data: snapshot resolution and download, the column contract, `try_cast` typing, price parsing,
de-duplication, the hash split, and the `TargetEncoder` fallback for unseen categories.

## 4. The numbers side by side (EUR MAE)

| | evaluation set | n | MAE |
|---|---|---:|---:|
| **Model v1 on A / B / A+B** | test split | 831 / 402 / 1,233 | **32.92 / 39.01 / 34.91** |
| Baseline (A+B medians) on C | C test split | 1,160 | 210.54 |
| Baseline built on C's own training split (C-local) | C test split | 1,160 | 111.07 |
| **Model v1 on C, first run** (week-1 outlier rule) | all priced C listings / C test split | 5,927 / 1,133 | **158.60 / 156.03** |
| Model v1 on C after pipeline fix P1 (103 expensive homes back in the evaluation set) | C test split | 1,160 | 171.82 |
| **Model v2 on C** (M1+M2; trained on A+B only, uses C's median price) | C test split | 1,160 | **108.75** |
| **Model v3 on C** (M1+M2+M3; trained on A+B+C train) | C test split | 1,160 | **82.89** |
| v2 / v3 on A+B (cost of the change at home) | A+B test split | 1,233 | 36.18 / 36.98 |

Relative error (MAPE) puts the euro numbers in proportion: v1 on A+B **23 %**, v1 on C **45 %**, v2 on C **41 %**, v3 on C **26 %**.

## 5. Explaining the gap

Going from €34.91 (A+B) to €171.82 (C, v1, same outlier rule) breaks down into four parts:

1. **Price level, about 60 € of the gap.** Multiply v1's predictions by the ratio of training medians (Amsterdam €285 / A+B €127.6 = 2.25)
   and the C MAE falls from €171.8 to **€112.7**. v2 does this inside the model and gets **€108.8**. Both are only as good as the C-local
   baseline (€111.1). Knowing Belgian price structure adds nothing in Amsterdam beyond what Amsterdam's own room-type and guest medians already say.
2. **Different price structure, about 26 €.** v2 over-prices private rooms (+€54 mean error) and hotel rooms, because one city-wide index is
   dominated by entire homes (81 %). In Amsterdam the entire-home premium is larger, and so is the distance-from-centre discount.
   Training on Amsterdam's own 80 % (v3) learns this: **€82.9**, 25 % better than the C-local baseline. In v3, `dist_centre_km`
   rises from the 7th to the 2nd most important feature.
3. **Proportional error, the rest.** v3's relative error on Amsterdam (26 %) is close to v1's on Brussels and Antwerp (23 %). The remaining
   euro gap (€82.9 vs €34.9) is mostly the same relative error on prices 2.2x higher. On top of that, Amsterdam has a larger expensive tail:
   listings above €700 average a €336 error and are under-priced by €311, like Belgium's above-€400 tail.
4. **What we cannot measure.** 43 % of Amsterdam listings have no price, so every C number covers only the active 57 %. The model still
   suggests a price for the blocked listings, but those suggestions can't be checked against anything.

## 6. Caveats we want to be explicit about

* **P1 makes v1 look worse**: €158.6 to €171.8. The first-run number was flattered by silently dropping the 103 most expensive real homes.
* **v3's own baseline is not comparable** to v1's: it pools medians across a 2.2x price gap, so its baseline MAE in Brussels jumps to €77.5,
  and its "improvement vs baseline" there is inflated. Judge v3 on C against the **C-local baseline (€111.1)**, and on A/B against v1's model MAE.
* **v3 costs a little at home**: Brussels €32.9 to €35.1, Antwerp €39.0 to €40.8. One pooled model trades some local fit for coverage.
  For production we would route Brussels/Antwerp to v1 and Amsterdam to v3, or retrain per city with the same code (it takes the city as a parameter).
* **v2 needs at least some priced listings in the new city** to compute the index. With none, it refuses to predict rather than guess.
* Single snapshot: quote dates vary per listing, so seasonality is not modelled, and Amsterdam's September quotes may sit near the seasonal peak.

## 7. What we learned about the pipeline

The first run did not crash, and that was the problem: the week-1 pipeline was robust to *format* changes (schema contract, `try_cast`,
URL resolution all worked on unseen data) but blind to *distribution* changes. The fixes that mattered most are not about parsing. They make
the pipeline say "this city does not look like the cities I was built on" (P2, P3), and they make the model learn something that transfers:
price relative to the city, not euros (M1, M2).
