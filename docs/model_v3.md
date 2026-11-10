# Price model v3

* **Model**: `hgb` (scikit-learn HistGradientBoostingRegressor on log(price / city price index), exp back), chosen by
  5-fold CV on the training split; the test split was not used for any choice.
* **Trained on**: brussels, antwerp, amsterdam: 9,671 listings (`split='train'`, `price_status='ok'`), trained at 2026-11-10T12:30:50 UTC.
* **Baseline**: median training price for the same (room_type, accommodates), falling back to the room-type median and then the global median.
* **Price rule**: 15.0 <= price <= 8.0 x city median (training and primary evaluation).
* **City price index** (median training-split price): {'amsterdam': 285.0, 'antwerp': 131.91500000000002, 'brussels': 125.0}

## Model choice: 5-fold CV MAE on the training split (EUR)

| model         |   mean |   std |
|:--------------|-------:|------:|
| hgb           |  60.93 |  2.2  |
| hgb_small     |  61.31 |  2.24 |
| random_forest |  62.35 |  2.59 |
| ridge         |  68.65 |  2.85 |
| baseline      | 114.73 |  4.13 |

## Test-set error (EUR), the numbers we report

| city                | evaluation_set   |    n |   mae_model |   mae_baseline |   improvement_vs_baseline |   median_abs_error_model |   mape_model |   mean_error_model |
|:--------------------|:-----------------|-----:|------------:|---------------:|--------------------------:|-------------------------:|-------------:|-------------------:|
| amsterdam           | test split       | 1160 |       82.89 |         145.96 |                    0.4321 |                    52.38 |       0.2598 |             -14.35 |
| antwerp             | test split       |  402 |       40.8  |          92.57 |                    0.5593 |                    25.6  |       0.2848 |               4.52 |
| brussels            | test split       |  831 |       35.13 |          77.54 |                    0.5469 |                    20.34 |       0.2152 |             -17.78 |
| ALL TRAINING CITIES | test split       | 2393 |       59.23 |         113.23 |                    0.4769 |                    34.42 |       0.2485 |             -12.37 |

`mean_error_model` < 0 means that on average the model predicts below the actual price. Because the model fits log(price), its
predictions sit near the median rather than the mean, so expensive listings are under-predicted in euro terms.

## Error by room type (test split)

|                                  |   n |   median_price |   mae_model |   mae_baseline |   mean_error |
|:---------------------------------|----:|---------------:|------------:|---------------:|-------------:|
| ('amsterdam', 'Entire home/apt') | 811 |          339.7 |        96.8 |          177.4 |        -21.2 |
| ('amsterdam', 'Hotel room')      |   6 |          155.5 |        84.8 |          145.6 |        -60.5 |
| ('amsterdam', 'Private room')    | 337 |          169   |        50.6 |           72.5 |          2.4 |
| ('amsterdam', 'Shared room')     |   6 |           61.5 |        18.7 |           19.1 |         10.3 |
| ('antwerp', 'Entire home/apt')   | 328 |          140.2 |        45.5 |          104.6 |          5.7 |
| ('antwerp', 'Hotel room')        |   1 |          149   |        14.4 |           21   |         14.4 |
| ('antwerp', 'Private room')      |  73 |           80   |        20.1 |           39.7 |         -1.1 |
| ('brussels', 'Entire home/apt')  | 595 |          139   |        38.2 |           89   |        -18.4 |
| ('brussels', 'Hotel room')       |   3 |          196   |        43.4 |           33.7 |        -43.4 |
| ('brussels', 'Private room')     | 227 |           93   |        27.6 |           49.7 |        -16   |
| ('brussels', 'Shared room')      |   6 |           32.5 |        13.6 |           17.2 |         -8.3 |

## Error by price band (test split)

| price_band   |   n |   mae_model |   mean_error |
|:-------------|----:|------------:|-------------:|
| (0, 75]      | 194 |        20.2 |         15.5 |
| (75, 125]    | 502 |        25.8 |         15.9 |
| (125, 200]   | 621 |        37.5 |          5.6 |
| (200, 400]   | 737 |        61.2 |         -1.3 |
| (400, 1000]  | 312 |       135.4 |        -96.4 |

The error grows with price: listings above €400 are rare, and the features do not capture what makes them expensive
(view, design, exact location).

## Which features matter: permutation importance on the test split

How much the test MAE rises when one feature is shuffled (5 repeats):

| feature                        |   MAE increase (EUR) |
|:-------------------------------|---------------------:|
| accommodates                   |                14.77 |
| dist_centre_km                 |                10.92 |
| bedrooms                       |                 9    |
| minimum_nights                 |                 6.09 |
| room_type                      |                 5.99 |
| nbhd_key                       |                 5.08 |
| bathrooms                      |                 2.93 |
| reviews_per_month              |                 2.39 |
| review_scores_cleanliness      |                 1.66 |
| number_of_reviews_ltm          |                 1.61 |
| calculated_host_listings_count |                 1.55 |
| amenities_count                |                 1.48 |
| review_scores_rating           |                 1.14 |
| review_scores_location         |                 0.96 |
| property_group                 |                 0.88 |
| is_long_stay                   |                 0.85 |
| number_of_reviews              |                 0.83 |
| hosts_time_as_host_years       |                 0.8  |
| review_scores_value            |                 0.69 |
| has_air_conditioning           |                 0.63 |
| bathroom_shared                |                 0.5  |
| beds                           |                 0.41 |
| has_washer                     |                 0.37 |
| has_elevator                   |                 0.26 |
| has_dishwasher                 |                 0.22 |
| has_free_parking               |                 0.16 |
| has_view                       |                 0.14 |
| has_self_checkin               |                 0.1  |
| host_is_superhost              |                 0.08 |
| has_hot_tub_or_pool            |                 0.07 |
| has_workspace                  |                 0.06 |
