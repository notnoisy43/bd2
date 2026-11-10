# Price model v1

* **Model**: `hgb_small` (scikit-learn HistGradientBoostingRegressor on log(price), exp back), chosen by
  5-fold CV on the training split; the test split was not used for any choice.
* **Trained on**: brussels, antwerp: 4,800 listings (`split='train'`, `price_status='ok'`), trained at 2026-11-10T12:15:28 UTC.
* **Baseline**: median training price for the same (room_type, accommodates), falling back to the room-type median and then the global median.
* **Price bounds**: training and primary evaluation use €15 to €1000.

## Model choice: 5-fold CV MAE on the training split (EUR)

| model         |   mean |   std |
|:--------------|-------:|------:|
| hgb_small     |  37.86 |  1.41 |
| hgb           |  38.05 |  1.24 |
| random_forest |  38.64 |  1.83 |
| ridge         |  42.9  |  1.72 |
| baseline      |  51.72 |  2.03 |

## Test-set error (EUR), the numbers we report

| city                | evaluation_set   |    n |   mae_model |   mae_baseline |   improvement_vs_baseline |   median_abs_error_model |   mape_model |   mean_error_model |
|:--------------------|:-----------------|-----:|------------:|---------------:|--------------------------:|-------------------------:|-------------:|-------------------:|
| antwerp             | test split       |  402 |       39.01 |          57.08 |                    0.3166 |                    22.19 |       0.2613 |              -3.73 |
| brussels            | test split       |  831 |       32.92 |          46.88 |                    0.2978 |                    20.49 |       0.2151 |              -7.17 |
| ALL TRAINING CITIES | test split       | 1233 |       34.91 |          50.21 |                    0.3047 |                    21    |       0.2302 |              -6.05 |

`mean_error_model` < 0 means that on average the model predicts below the actual price. Because the model fits log(price), its
predictions sit near the median rather than the mean, so expensive listings are under-predicted in euro terms.

## Error by room type (test split)

|                                 |   n |   median_price |   mae_model |   mae_baseline |   mean_error |
|:--------------------------------|----:|---------------:|------------:|---------------:|-------------:|
| ('antwerp', 'Entire home/apt')  | 328 |          140.2 |        43.2 |           64.5 |         -4.5 |
| ('antwerp', 'Hotel room')       |   1 |          149   |        23.2 |           21.5 |         23.2 |
| ('antwerp', 'Private room')     |  73 |           80   |        20.6 |           24.1 |         -0.5 |
| ('brussels', 'Entire home/apt') | 595 |          139   |        35.8 |           49.8 |         -7.2 |
| ('brussels', 'Hotel room')      |   3 |          196   |        30.2 |           33.2 |        -27.3 |
| ('brussels', 'Private room')    | 227 |           93   |        26   |           40.1 |         -7   |
| ('brussels', 'Shared room')     |   6 |           32.5 |         8.6 |           20.2 |         -2.3 |

## Error by price band (test split)

| price_band   |   n |   mae_model |   mean_error |
|:-------------|----:|------------:|-------------:|
| (0, 75]      | 180 |        19.2 |         15.2 |
| (75, 125]    | 420 |        21   |         12.8 |
| (125, 200]   | 417 |        27.6 |         -4   |
| (200, 400]   | 174 |        69.1 |        -44.9 |
| (400, 1000]  |  42 |       172.8 |       -144.6 |

The error grows with price: listings above €400 are rare, and the features do not capture what makes them expensive
(view, design, exact location).

## Which features matter: permutation importance on the test split

How much the test MAE rises when one feature is shuffled (5 repeats):

| feature                        |   MAE increase (EUR) |
|:-------------------------------|---------------------:|
| accommodates                   |                 9.75 |
| bedrooms                       |                 7.07 |
| minimum_nights                 |                 6.87 |
| city                           |                 2.83 |
| room_type                      |                 2.5  |
| bathrooms                      |                 2.21 |
| dist_centre_km                 |                 2.01 |
| is_long_stay                   |                 1.82 |
| reviews_per_month              |                 1.73 |
| property_group                 |                 1.21 |
| calculated_host_listings_count |                 0.99 |
| review_scores_rating           |                 0.94 |
| nbhd_key                       |                 0.92 |
| amenities_count                |                 0.81 |
| beds                           |                 0.75 |
| review_scores_value            |                 0.75 |
| bathroom_shared                |                 0.71 |
| number_of_reviews              |                 0.66 |
| number_of_reviews_ltm          |                 0.57 |
| review_scores_cleanliness      |                 0.53 |
| review_scores_location         |                 0.45 |
| hosts_time_as_host_years       |                 0.44 |
| has_air_conditioning           |                 0.38 |
| has_free_parking               |                 0.34 |
| has_washer                     |                 0.28 |
| has_hot_tub_or_pool            |                 0.23 |
| has_elevator                   |                 0.17 |
| has_dishwasher                 |                 0.13 |
| has_self_checkin               |                 0.13 |
| host_is_superhost              |                 0.02 |
| has_view                       |                 0    |
| has_workspace                  |                -0.02 |
