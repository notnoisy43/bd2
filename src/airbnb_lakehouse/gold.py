"""Gold: per-city metrics for the dashboard/Genie and listing-level model features."""
from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

from . import config
from .lakehouse import Lakehouse

# amenity flag -> SQL LIKE pattern on lower(amenity)
AMENITY_FLAGS = {
    "has_air_conditioning": "%air conditioning%",
    "has_dishwasher": "%dishwasher%",
    "has_washer": "%washer%",
    "has_free_parking": "%free%parking%",
    "has_elevator": "%elevator%",
    "has_self_checkin": "%self check-in%",
    "has_workspace": "%workspace%",
    "has_view": "%view%",
    "has_hot_tub_or_pool": "%hot tub%",
}


def _amenity(pattern: str) -> F.Column:
    return F.coalesce(F.expr(f"exists(amenities, a -> lower(a) LIKE '{pattern}')"), F.lit(False))


def build_listing_features(silver: DataFrame) -> DataFrame:
    """One row per listing.  Only columns that are known *without* the price and that are not
    consequences of the price are kept as features (see EDA notebook, leakage review)."""
    centre = silver.agg(F.percentile_approx("latitude", 0.5).alias("c_lat"),
                        F.percentile_approx("longitude", 0.5).alias("c_lon"))
    df = silver.crossJoin(F.broadcast(centre))
    # haversine distance (km) to the city's listing-density centre (median lat/lon): a
    # geography feature that means the same thing in every city, unlike raw coordinates.
    dist = 2 * 6371.0 * F.asin(F.sqrt(
        F.pow(F.sin(F.radians(F.col("latitude") - F.col("c_lat")) / 2), 2)
        + F.cos(F.radians("c_lat")) * F.cos(F.radians("latitude"))
        * F.pow(F.sin(F.radians(F.col("longitude") - F.col("c_lon")) / 2), 2)))

    split = F.when(F.pmod(F.xxhash64(F.col("id"), F.lit(config.SPLIT_SEED)), F.lit(100))
                   < config.TEST_PCT, "test").otherwise("train")

    return df.select(
        "city", "snapshot_date", F.col("id").alias("listing_id"), "name", "listing_url",
        "neighbourhood_cleansed", "latitude", "longitude",
        # target + flags
        "price", "price_status", "is_active",
        split.alias("split"),
        # features
        "room_type", "property_base", "accommodates", "bedrooms", "beds", "bathrooms",
        F.col("bathroom_shared").cast("int").alias("bathroom_shared"),
        "amenities_count",
        *[_amenity(p).cast("int").alias(n) for n, p in AMENITY_FLAGS.items()],
        F.least(F.col("minimum_nights"), F.lit(365)).alias("minimum_nights"),
        F.col("is_long_stay").cast("int").alias("is_long_stay"),
        "number_of_reviews", "number_of_reviews_ltm", "reviews_per_month",
        "review_scores_rating", "review_scores_cleanliness", "review_scores_location",
        "review_scores_value",
        F.col("host_is_superhost").cast("int").alias("host_is_superhost"),
        "calculated_host_listings_count", "hosts_time_as_host_years",
        F.round(dist, 3).alias("dist_centre_km"),
    )


def build_city_metrics(silver: DataFrame) -> DataFrame:
    priced = F.col("price_status") == "ok"
    return silver.groupBy("city", "snapshot_date").agg(
        F.count("*").alias("listings"),
        F.sum(F.col("is_active").cast("int")).alias("active_listings"),
        F.sum(priced.cast("int")).alias("priced_listings"),
        F.countDistinct("host_id").alias("hosts"),
        F.round(F.percentile_approx(F.when(priced, F.col("price")), 0.5), 2).alias("median_price"),
        F.round(F.avg(F.when(priced, F.col("price"))), 2).alias("mean_price"),
        F.round(F.percentile_approx(F.when(priced, F.col("price")), 0.25), 2).alias("p25_price"),
        F.round(F.percentile_approx(F.when(priced, F.col("price")), 0.75), 2).alias("p75_price"),
        F.round(F.avg("availability_365"), 1).alias("avg_availability_365"),
        F.round(F.avg((F.col("availability_365") == 0).cast("int")), 4).alias("share_unavailable_365"),
        F.round(F.avg("review_scores_rating"), 3).alias("avg_review_rating"),
        F.round(F.avg((F.col("room_type") == "Entire home/apt").cast("int")), 4).alias("share_entire_home"),
        F.round(F.avg(F.col("is_multi_listing_host").cast("int")), 4).alias("share_multi_listing_host"),
        F.round(F.avg(F.col("host_is_superhost").cast("int")), 4).alias("share_superhost"),
        F.round(F.avg(F.col("is_long_stay").cast("int")), 4).alias("share_long_stay"),
        F.round(F.percentile_approx("estimated_revenue_l365d", 0.5), 0).alias("median_est_revenue_l365d"),
        F.lit(config.CURRENCY).alias("currency"),
    )


def build_neighbourhood_metrics(silver: DataFrame) -> DataFrame:
    priced = F.col("price_status") == "ok"
    return silver.groupBy("city", F.col("neighbourhood_cleansed").alias("neighbourhood")).agg(
        F.count("*").alias("listings"),
        F.sum(priced.cast("int")).alias("priced_listings"),
        F.round(F.percentile_approx(F.when(priced, F.col("price")), 0.5), 2).alias("median_price"),
        F.round(F.avg("review_scores_rating"), 3).alias("avg_review_rating"),
        F.round(F.avg("availability_365"), 1).alias("avg_availability_365"),
        F.round(F.avg((F.col("room_type") == "Entire home/apt").cast("int")), 4).alias("share_entire_home"),
        F.round(F.avg("latitude"), 5).alias("lat"), F.round(F.avg("longitude"), 5).alias("lon"),
    )


def build_room_type_metrics(silver: DataFrame) -> DataFrame:
    priced = F.col("price_status") == "ok"
    total = silver.count()
    return silver.groupBy("city", "room_type").agg(
        F.count("*").alias("listings"),
        F.round(F.count("*") / F.lit(total), 4).alias("share_of_city"),
        F.round(F.percentile_approx(F.when(priced, F.col("price")), 0.5), 2).alias("median_price"),
        F.round(F.avg("accommodates"), 2).alias("avg_accommodates"),
    )


def run_gold(lh: Lakehouse, city: str) -> dict:
    silver = lh.read("silver_listings").where(F.col("city") == city)
    out = {
        "gold_listing_features": build_listing_features(silver),
        "gold_city_metrics": build_city_metrics(silver),
        "gold_neighbourhood_metrics": build_neighbourhood_metrics(silver),
        "gold_room_type_metrics": build_room_type_metrics(silver),
    }
    counts = {name: lh.write_city(df, name, city) for name, df in out.items()}
    print(f"[gold] {city}: {counts}")
    return {"city": city, **counts}
