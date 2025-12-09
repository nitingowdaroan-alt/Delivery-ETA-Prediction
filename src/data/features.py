"""
Feature Engineering Module

Creates predictive features for delivery ETA prediction:
1. Geographic features (haversine, bearing, zone encodings)
2. Temporal features (hour, day, cyclical encodings)
3. Aggregated features (zone stats, restaurant stats, rider stats)
4. Target encoding for categorical variables
5. Lag features for time-series patterns

Design Philosophy:
- All transformations are fit on training data only (no leakage)
- Features are designed to be computed in real-time for inference
- Scikit-learn compatible transformers for pipeline integration
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger.add("logs/features.log", rotation="10 MB")


class GeoFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Geographic feature engineering.

    Creates:
    - Haversine distance (if not present)
    - Bearing (compass direction)
    - Manhattan-like distance proxy
    - Zone centroid distances
    """

    def __init__(self, zone_centroids: Optional[Dict[int, Tuple[float, float]]] = None):
        """
        Initialize transformer.

        Args:
            zone_centroids: Dict mapping zone_id to (lat, lng) centroids
        """
        self.zone_centroids = zone_centroids or {}

    def fit(self, X: pd.DataFrame, y=None):
        """Fit: compute zone centroids from data if not provided."""
        if not self.zone_centroids and "zone_id" in X.columns:
            self.zone_centroids = (
                X.groupby("zone_id")
                .agg({"pickup_lat": "mean", "pickup_lng": "mean"})
                .apply(lambda row: (row["pickup_lat"], row["pickup_lng"]), axis=1)
                .to_dict()
            )
            logger.info(f"Computed centroids for {len(self.zone_centroids)} zones")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform: add geographic features."""
        X = X.copy()

        # Haversine distance (if not already computed)
        if "haversine_distance_km" not in X.columns:
            X["haversine_distance_km"] = X.apply(
                lambda row: self._haversine(
                    row["pickup_lat"], row["pickup_lng"],
                    row["dropoff_lat"], row["dropoff_lng"]
                ),
                axis=1
            )

        # Bearing (direction from pickup to dropoff)
        X["bearing"] = X.apply(
            lambda row: self._bearing(
                row["pickup_lat"], row["pickup_lng"],
                row["dropoff_lat"], row["dropoff_lng"]
            ),
            axis=1
        )

        # Bearing as cyclical features (sin/cos encoding)
        X["bearing_sin"] = np.sin(np.radians(X["bearing"]))
        X["bearing_cos"] = np.cos(np.radians(X["bearing"]))

        # Lat/lng differences (proxy for Manhattan distance in grid cities)
        X["lat_diff"] = abs(X["dropoff_lat"] - X["pickup_lat"])
        X["lng_diff"] = abs(X["dropoff_lng"] - X["pickup_lng"])
        X["manhattan_proxy"] = X["lat_diff"] + X["lng_diff"]

        # Distance to zone centroid (useful for zone-level patterns)
        if "zone_id" in X.columns and self.zone_centroids:
            X["dist_to_zone_centroid"] = X.apply(
                lambda row: self._dist_to_centroid(row), axis=1
            )

        return X

    def _haversine(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate haversine distance in km."""
        try:
            return geodesic((lat1, lng1), (lat2, lng2)).kilometers
        except Exception:
            return 0.0

    def _bearing(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate bearing (compass direction) from point 1 to point 2."""
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        diff_lng = math.radians(lng2 - lng1)

        x = math.sin(diff_lng) * math.cos(lat2_r)
        y = math.cos(lat1_r) * math.sin(lat2_r) - (
            math.sin(lat1_r) * math.cos(lat2_r) * math.cos(diff_lng)
        )

        bearing = math.atan2(x, y)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360

        return bearing

    def _dist_to_centroid(self, row: pd.Series) -> float:
        """Calculate distance from pickup to zone centroid."""
        zone_id = row.get("zone_id")
        if zone_id in self.zone_centroids:
            centroid = self.zone_centroids[zone_id]
            return self._haversine(
                row["pickup_lat"], row["pickup_lng"],
                centroid[0], centroid[1]
            )
        return 0.0


class TemporalFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Temporal feature engineering.

    Creates:
    - Hour, day, week, month features
    - Cyclical encodings (sin/cos)
    - Rush hour indicators
    - Holiday/special event indicators
    """

    def __init__(self, add_cyclical: bool = True):
        """
        Initialize transformer.

        Args:
            add_cyclical: Whether to add sin/cos cyclical features
        """
        self.add_cyclical = add_cyclical

    def fit(self, X: pd.DataFrame, y=None):
        """No fitting needed for temporal features."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform: add temporal features."""
        X = X.copy()

        # Extract from timestamp if available
        if "order_timestamp" in X.columns:
            ts = pd.to_datetime(X["order_timestamp"])

            if "hour_of_day" not in X.columns:
                X["hour_of_day"] = ts.dt.hour
            if "day_of_week" not in X.columns:
                X["day_of_week"] = ts.dt.dayofweek
            if "month" not in X.columns:
                X["month"] = ts.dt.month
            if "day_of_month" not in X.columns:
                X["day_of_month"] = ts.dt.day
            if "week_of_year" not in X.columns:
                X["week_of_year"] = ts.dt.isocalendar().week.astype(int)

        # Cyclical encoding for hour (captures midnight wrap-around)
        if self.add_cyclical and "hour_of_day" in X.columns:
            X["hour_sin"] = np.sin(2 * np.pi * X["hour_of_day"] / 24)
            X["hour_cos"] = np.cos(2 * np.pi * X["hour_of_day"] / 24)

        # Cyclical encoding for day of week
        if self.add_cyclical and "day_of_week" in X.columns:
            X["dow_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
            X["dow_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)

        # Cyclical encoding for month
        if self.add_cyclical and "month" in X.columns:
            X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
            X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

        # Rush hour indicators (if not already present)
        if "hour_of_day" in X.columns:
            if "is_morning_rush" not in X.columns:
                X["is_morning_rush"] = X["hour_of_day"].between(7, 9).astype(int)
            if "is_lunch_rush" not in X.columns:
                X["is_lunch_rush"] = X["hour_of_day"].between(11, 14).astype(int)
            if "is_dinner_rush" not in X.columns:
                X["is_dinner_rush"] = X["hour_of_day"].between(17, 21).astype(int)
            if "is_late_night" not in X.columns:
                X["is_late_night"] = ((X["hour_of_day"] >= 22) | (X["hour_of_day"] <= 5)).astype(int)

        # Weekend indicator
        if "day_of_week" in X.columns and "is_weekend" not in X.columns:
            X["is_weekend"] = (X["day_of_week"] >= 5).astype(int)

        return X


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoding for categorical variables.

    Why target encoding?
    - Handles high-cardinality categoricals (hundreds of restaurants/zones)
    - Captures relationship between category and target
    - Regularization prevents overfitting to rare categories

    Uses smoothing: encoded = (n * category_mean + m * global_mean) / (n + m)
    where n = category count, m = smoothing parameter
    """

    def __init__(
        self,
        columns: List[str],
        smoothing: float = 10.0,
        min_samples: int = 5,
    ):
        """
        Initialize target encoder.

        Args:
            columns: Columns to encode
            smoothing: Smoothing parameter (higher = more regularization)
            min_samples: Minimum samples for a category to be encoded
        """
        self.columns = columns
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.encodings: Dict[str, Dict[Any, float]] = {}
        self.global_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit: compute target encodings for each category."""
        self.global_mean = y.mean()

        for col in self.columns:
            if col not in X.columns:
                continue

            # Group by category, compute mean and count
            df_temp = pd.DataFrame({"category": X[col], "target": y})
            agg = df_temp.groupby("category")["target"].agg(["mean", "count"])

            # Apply smoothing
            smoothed = (
                agg["count"] * agg["mean"] + self.smoothing * self.global_mean
            ) / (agg["count"] + self.smoothing)

            # Filter by minimum samples
            smoothed = smoothed[agg["count"] >= self.min_samples]

            self.encodings[col] = smoothed.to_dict()
            logger.info(f"Target encoded {col}: {len(self.encodings[col])} categories")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform: apply target encodings."""
        X = X.copy()

        for col in self.columns:
            if col not in X.columns or col not in self.encodings:
                continue

            new_col = f"{col}_encoded"
            X[new_col] = X[col].map(self.encodings[col]).fillna(self.global_mean)

        return X


class AggregatedFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Creates aggregated/statistical features.

    Computes zone-level and restaurant-level statistics:
    - Mean/median ETA per zone
    - Order volume per zone/restaurant
    - Historical performance metrics
    """

    def __init__(self, agg_columns: Optional[List[str]] = None):
        """
        Initialize transformer.

        Args:
            agg_columns: Columns to aggregate on (default: zone_id, restaurant_id)
        """
        self.agg_columns = agg_columns or ["zone_id", "restaurant_id"]
        self.aggregations: Dict[str, pd.DataFrame] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit: compute aggregations on training data."""
        df_temp = X.copy()
        df_temp["target"] = y

        for col in self.agg_columns:
            if col not in X.columns:
                continue

            agg = df_temp.groupby(col).agg({
                "target": ["mean", "median", "std", "count"],
            })
            agg.columns = [
                f"{col}_eta_mean",
                f"{col}_eta_median",
                f"{col}_eta_std",
                f"{col}_order_count",
            ]
            agg = agg.reset_index()

            self.aggregations[col] = agg
            logger.info(f"Computed aggregations for {col}: {len(agg)} groups")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform: merge aggregated features."""
        X = X.copy()

        for col, agg_df in self.aggregations.items():
            if col in X.columns:
                X = X.merge(agg_df, on=col, how="left")

                # Fill missing values with global statistics
                for agg_col in agg_df.columns:
                    if agg_col != col and agg_col in X.columns:
                        if "mean" in agg_col or "median" in agg_col:
                            X[agg_col] = X[agg_col].fillna(agg_df[agg_col].mean())
                        elif "std" in agg_col:
                            X[agg_col] = X[agg_col].fillna(agg_df[agg_col].median())
                        elif "count" in agg_col:
                            X[agg_col] = X[agg_col].fillna(0)

        return X


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Creates lag features for time-series patterns.

    WARNING: Be careful with lag features - they can cause data leakage
    if not handled properly. Only use lagged values that would be
    available at prediction time.
    """

    def __init__(
        self,
        group_cols: List[str],
        lag_col: str = "actual_eta_minutes",
        lags: List[int] = None,
    ):
        """
        Initialize transformer.

        Args:
            group_cols: Columns to group by (e.g., zone_id, restaurant_id)
            lag_col: Column to create lags for
            lags: List of lag periods (in rows)
        """
        self.group_cols = group_cols
        self.lag_col = lag_col
        self.lags = lags or [1, 3, 7]  # Last 1, 3, 7 orders
        self.lag_stats: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Fit: store global statistics for filling missing lags."""
        if self.lag_col in X.columns:
            self.lag_stats["mean"] = X[self.lag_col].mean()
            self.lag_stats["median"] = X[self.lag_col].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform: add lag features (grouped by specified columns)."""
        X = X.copy()

        # Sort by timestamp if available
        if "order_timestamp" in X.columns:
            X = X.sort_values("order_timestamp")

        if self.lag_col not in X.columns:
            return X

        for col in self.group_cols:
            if col not in X.columns:
                continue

            for lag in self.lags:
                lag_name = f"{col}_{self.lag_col}_lag_{lag}"
                X[lag_name] = X.groupby(col)[self.lag_col].shift(lag)
                X[lag_name] = X[lag_name].fillna(self.lag_stats.get("mean", 30))

            # Rolling mean
            roll_name = f"{col}_{self.lag_col}_rolling_mean"
            X[roll_name] = X.groupby(col)[self.lag_col].transform(
                lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
            )
            X[roll_name] = X[roll_name].fillna(self.lag_stats.get("mean", 30))

        return X


class WeatherFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Weather feature encoding.

    Converts weather conditions to numeric features.
    Can be extended to pull real-time weather data.
    """

    WEATHER_ENCODING = {
        "clear": 0,
        "cloudy": 1,
        "fog": 2,
        "rain": 3,
        "heavy_rain": 4,
    }

    def fit(self, X: pd.DataFrame, y=None):
        """No fitting needed."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform: encode weather conditions."""
        X = X.copy()

        if "weather_condition" in X.columns:
            X["weather_encoded"] = X["weather_condition"].map(self.WEATHER_ENCODING).fillna(0)

            # One-hot encode for interpretability
            weather_dummies = pd.get_dummies(
                X["weather_condition"],
                prefix="weather",
                drop_first=True
            )
            X = pd.concat([X, weather_dummies], axis=1)

        return X


class FeaturePipeline:
    """
    Complete feature engineering pipeline.

    Combines all transformers in proper order.
    Handles fit/transform semantics for training vs inference.
    """

    def __init__(
        self,
        target_encode_cols: Optional[List[str]] = None,
        agg_cols: Optional[List[str]] = None,
        add_lag_features: bool = False,
    ):
        """
        Initialize feature pipeline.

        Args:
            target_encode_cols: Columns for target encoding
            agg_cols: Columns for aggregation features
            add_lag_features: Whether to add lag features (careful with leakage)
        """
        self.target_encode_cols = target_encode_cols or ["zone_id", "restaurant_type"]
        self.agg_cols = agg_cols or ["zone_id"]
        self.add_lag_features = add_lag_features

        # Initialize transformers
        self.geo_transformer = GeoFeatureTransformer()
        self.temporal_transformer = TemporalFeatureTransformer()
        self.target_encoder = TargetEncoder(columns=self.target_encode_cols)
        self.agg_transformer = AggregatedFeatureTransformer(agg_columns=self.agg_cols)
        self.weather_transformer = WeatherFeatureTransformer()
        self.lag_transformer = LagFeatureTransformer(group_cols=["zone_id"])

        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeaturePipeline":
        """Fit all transformers on training data."""
        logger.info("Fitting feature pipeline...")

        # Fit each transformer
        self.geo_transformer.fit(X)
        self.temporal_transformer.fit(X)
        self.target_encoder.fit(X, y)
        self.agg_transformer.fit(X, y)
        self.weather_transformer.fit(X)

        if self.add_lag_features:
            self.lag_transformer.fit(X)

        self._is_fitted = True
        logger.info("Feature pipeline fitted")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted transformers."""
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        logger.info(f"Transforming {len(X)} rows...")

        # Apply transformers in order
        X = self.geo_transformer.transform(X)
        X = self.temporal_transformer.transform(X)
        X = self.target_encoder.transform(X)
        X = self.agg_transformer.transform(X)
        X = self.weather_transformer.transform(X)

        if self.add_lag_features:
            X = self.lag_transformer.transform(X)

        logger.info(f"Transformation complete. Shape: {X.shape}")
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names(self) -> List[str]:
        """Get list of engineered feature names."""
        return [
            # Geo features
            "haversine_distance_km",
            "bearing",
            "bearing_sin",
            "bearing_cos",
            "lat_diff",
            "lng_diff",
            "manhattan_proxy",
            "dist_to_zone_centroid",
            # Temporal features
            "hour_of_day",
            "day_of_week",
            "month",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "is_morning_rush",
            "is_lunch_rush",
            "is_dinner_rush",
            "is_late_night",
            "is_weekend",
            # Target encoded
            "zone_id_encoded",
            "restaurant_type_encoded",
            # Aggregated
            "zone_id_eta_mean",
            "zone_id_eta_median",
            "zone_id_eta_std",
            "zone_id_order_count",
            # Weather
            "weather_encoded",
            "weather_multiplier",
            # Raw features
            "traffic_multiplier",
            "rider_avg_speed_kmh",
            "rider_reliability",
            "rider_completed_deliveries",
            "prep_time_minutes",
        ]


def prepare_features_for_training(
    df: pd.DataFrame,
    target_col: str = "actual_eta_minutes",
) -> Tuple[pd.DataFrame, pd.Series, FeaturePipeline]:
    """
    Prepare features for model training.

    Args:
        df: Cleaned dataframe
        target_col: Name of target column

    Returns:
        (X, y, fitted_pipeline)
    """
    # Separate features and target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col], errors="ignore")

    # Initialize and fit pipeline
    pipeline = FeaturePipeline(
        target_encode_cols=["zone_id", "restaurant_type"],
        agg_cols=["zone_id", "restaurant_id"],
        add_lag_features=False,  # Disable for training to avoid leakage
    )

    X_transformed = pipeline.fit_transform(X, y)

    logger.info(f"Prepared {len(X_transformed)} samples with {X_transformed.shape[1]} features")

    return X_transformed, y, pipeline


def prepare_features_for_inference(
    df: pd.DataFrame,
    pipeline: FeaturePipeline,
) -> pd.DataFrame:
    """
    Prepare features for real-time inference.

    Args:
        df: Raw input data (single row or batch)
        pipeline: Fitted feature pipeline

    Returns:
        Transformed features ready for model prediction
    """
    return pipeline.transform(df)


# Feature columns used for model training (excludes IDs and target)
MODEL_FEATURES = [
    "haversine_distance_km",
    "bearing_sin",
    "bearing_cos",
    "lat_diff",
    "lng_diff",
    "manhattan_proxy",
    "hour_of_day",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_morning_rush",
    "is_lunch_rush",
    "is_dinner_rush",
    "is_late_night",
    "is_weekend",
    "zone_id_encoded",
    "restaurant_type_encoded",
    "zone_id_eta_mean",
    "zone_id_eta_std",
    "zone_id_order_count",
    "weather_encoded",
    "weather_multiplier",
    "traffic_multiplier",
    "rider_avg_speed_kmh",
    "rider_reliability",
    "prep_time_minutes",
]
