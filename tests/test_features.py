"""
Unit Tests for Feature Engineering Module

Tests:
1. Geographic feature calculations (haversine, bearing)
2. Temporal feature extraction
3. Target encoding
4. Aggregated features
5. Full pipeline integration
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime


# Import modules under test
import sys
sys.path.insert(0, ".")
from src.data.features import (
    GeoFeatureTransformer,
    TemporalFeatureTransformer,
    TargetEncoder,
    AggregatedFeatureTransformer,
    FeaturePipeline,
    MODEL_FEATURES,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_data():
    """Create sample order data for testing."""
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame({
        "order_id": [f"O{i:05d}" for i in range(n_samples)],
        "order_timestamp": pd.date_range("2023-01-01", periods=n_samples, freq="H"),
        "pickup_lat": np.random.uniform(37.7, 37.8, n_samples),
        "pickup_lng": np.random.uniform(-122.5, -122.4, n_samples),
        "dropoff_lat": np.random.uniform(37.7, 37.8, n_samples),
        "dropoff_lng": np.random.uniform(-122.5, -122.4, n_samples),
        "zone_id": np.random.randint(1, 5, n_samples),
        "restaurant_type": np.random.choice(["fast_food", "casual", "fine_dining"], n_samples),
        "restaurant_id": [f"R{i % 10:04d}" for i in range(n_samples)],
        "traffic_multiplier": np.random.uniform(0.8, 1.5, n_samples),
        "weather_condition": np.random.choice(["clear", "rain", "cloudy"], n_samples),
        "weather_multiplier": np.random.uniform(1.0, 1.3, n_samples),
        "prep_time_minutes": np.random.uniform(5, 25, n_samples),
        "rider_avg_speed_kmh": np.random.uniform(15, 35, n_samples),
        "rider_reliability": np.random.uniform(0.8, 1.0, n_samples),
    })

    return data


@pytest.fixture
def sample_target():
    """Create sample target variable."""
    np.random.seed(42)
    return pd.Series(np.random.uniform(10, 50, 100), name="actual_eta_minutes")


# =============================================================================
# GeoFeatureTransformer Tests
# =============================================================================

class TestGeoFeatureTransformer:
    """Tests for geographic feature engineering."""

    def test_haversine_distance_calculation(self):
        """Test haversine distance is calculated correctly."""
        transformer = GeoFeatureTransformer()

        # Known distance: SF to Oakland ~12 km
        data = pd.DataFrame({
            "pickup_lat": [37.7749],
            "pickup_lng": [-122.4194],
            "dropoff_lat": [37.8044],
            "dropoff_lng": [-122.2712],
        })

        transformer.fit(data)
        result = transformer.transform(data)

        assert "haversine_distance_km" in result.columns
        # Distance should be approximately 12-14 km
        assert 10 < result["haversine_distance_km"].iloc[0] < 16

    def test_bearing_calculation(self):
        """Test bearing is calculated and within valid range."""
        transformer = GeoFeatureTransformer()

        data = pd.DataFrame({
            "pickup_lat": [37.7749],
            "pickup_lng": [-122.4194],
            "dropoff_lat": [37.8044],
            "dropoff_lng": [-122.2712],
        })

        transformer.fit(data)
        result = transformer.transform(data)

        assert "bearing" in result.columns
        assert 0 <= result["bearing"].iloc[0] < 360

    def test_cyclical_bearing_features(self):
        """Test sin/cos bearing features are created."""
        transformer = GeoFeatureTransformer()
        data = pd.DataFrame({
            "pickup_lat": [37.77],
            "pickup_lng": [-122.42],
            "dropoff_lat": [37.78],
            "dropoff_lng": [-122.41],
        })

        transformer.fit(data)
        result = transformer.transform(data)

        assert "bearing_sin" in result.columns
        assert "bearing_cos" in result.columns
        # Sin and cos should be in [-1, 1]
        assert -1 <= result["bearing_sin"].iloc[0] <= 1
        assert -1 <= result["bearing_cos"].iloc[0] <= 1

    def test_manhattan_proxy(self, sample_data):
        """Test Manhattan distance proxy calculation."""
        transformer = GeoFeatureTransformer()
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)

        assert "lat_diff" in result.columns
        assert "lng_diff" in result.columns
        assert "manhattan_proxy" in result.columns
        assert (result["manhattan_proxy"] >= 0).all()


# =============================================================================
# TemporalFeatureTransformer Tests
# =============================================================================

class TestTemporalFeatureTransformer:
    """Tests for temporal feature engineering."""

    def test_hour_extraction(self, sample_data):
        """Test hour of day is extracted correctly."""
        transformer = TemporalFeatureTransformer()
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)

        assert "hour_of_day" in result.columns
        assert result["hour_of_day"].min() >= 0
        assert result["hour_of_day"].max() <= 23

    def test_day_of_week_extraction(self, sample_data):
        """Test day of week is extracted correctly."""
        transformer = TemporalFeatureTransformer()
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)

        assert "day_of_week" in result.columns
        assert result["day_of_week"].min() >= 0
        assert result["day_of_week"].max() <= 6

    def test_cyclical_hour_features(self, sample_data):
        """Test cyclical hour encoding."""
        transformer = TemporalFeatureTransformer(add_cyclical=True)
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)

        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        # All values should be in [-1, 1]
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()

    def test_rush_hour_indicators(self, sample_data):
        """Test rush hour indicator flags."""
        transformer = TemporalFeatureTransformer()
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)

        assert "is_lunch_rush" in result.columns
        assert "is_dinner_rush" in result.columns
        assert "is_morning_rush" in result.columns
        assert result["is_lunch_rush"].isin([0, 1]).all()

    def test_weekend_indicator(self, sample_data):
        """Test weekend indicator."""
        transformer = TemporalFeatureTransformer()
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)

        assert "is_weekend" in result.columns


# =============================================================================
# TargetEncoder Tests
# =============================================================================

class TestTargetEncoder:
    """Tests for target encoding."""

    def test_target_encoding_fit(self, sample_data, sample_target):
        """Test target encoder fitting."""
        encoder = TargetEncoder(columns=["zone_id"])
        encoder.fit(sample_data, sample_target)

        assert "zone_id" in encoder.encodings
        assert len(encoder.encodings["zone_id"]) > 0

    def test_target_encoding_transform(self, sample_data, sample_target):
        """Test target encoder transform creates encoded column."""
        encoder = TargetEncoder(columns=["zone_id"])
        encoder.fit(sample_data, sample_target)
        result = encoder.transform(sample_data)

        assert "zone_id_encoded" in result.columns
        # Encoded values should be numeric
        assert result["zone_id_encoded"].dtype in [np.float64, np.float32]

    def test_target_encoding_unseen_category(self, sample_data, sample_target):
        """Test handling of unseen categories."""
        encoder = TargetEncoder(columns=["zone_id"])
        encoder.fit(sample_data, sample_target)

        # Create new data with unseen zone_id
        new_data = sample_data.head(1).copy()
        new_data["zone_id"] = 999  # Unseen zone

        result = encoder.transform(new_data)

        # Should use global mean for unseen category
        assert result["zone_id_encoded"].iloc[0] == pytest.approx(encoder.global_mean, rel=0.01)

    def test_smoothing_effect(self, sample_data, sample_target):
        """Test that smoothing affects encoding."""
        encoder_low = TargetEncoder(columns=["zone_id"], smoothing=1)
        encoder_high = TargetEncoder(columns=["zone_id"], smoothing=100)

        encoder_low.fit(sample_data, sample_target)
        encoder_high.fit(sample_data, sample_target)

        # High smoothing should pull values closer to global mean
        result_low = encoder_low.transform(sample_data)
        result_high = encoder_high.transform(sample_data)

        low_var = result_low["zone_id_encoded"].var()
        high_var = result_high["zone_id_encoded"].var()

        # High smoothing should have lower variance
        assert high_var < low_var


# =============================================================================
# AggregatedFeatureTransformer Tests
# =============================================================================

class TestAggregatedFeatureTransformer:
    """Tests for aggregated features."""

    def test_zone_aggregations(self, sample_data, sample_target):
        """Test zone-level aggregations are computed."""
        transformer = AggregatedFeatureTransformer(agg_columns=["zone_id"])
        transformer.fit(sample_data, sample_target)
        result = transformer.transform(sample_data)

        assert "zone_id_eta_mean" in result.columns
        assert "zone_id_eta_median" in result.columns
        assert "zone_id_eta_std" in result.columns
        assert "zone_id_order_count" in result.columns

    def test_aggregation_values(self, sample_data, sample_target):
        """Test aggregation values are reasonable."""
        transformer = AggregatedFeatureTransformer(agg_columns=["zone_id"])
        transformer.fit(sample_data, sample_target)
        result = transformer.transform(sample_data)

        # Mean should be within target range
        assert result["zone_id_eta_mean"].min() > 0
        # Std should be non-negative
        assert (result["zone_id_eta_std"] >= 0).all()


# =============================================================================
# FeaturePipeline Integration Tests
# =============================================================================

class TestFeaturePipeline:
    """Integration tests for full feature pipeline."""

    def test_pipeline_fit_transform(self, sample_data, sample_target):
        """Test full pipeline fit and transform."""
        pipeline = FeaturePipeline(
            target_encode_cols=["zone_id"],
            agg_cols=["zone_id"],
            add_lag_features=False,
        )

        result = pipeline.fit_transform(sample_data, sample_target)

        # Should have more columns than original
        assert result.shape[1] > sample_data.shape[1]

    def test_pipeline_reusability(self, sample_data, sample_target):
        """Test pipeline can be reused for transform after fit."""
        pipeline = FeaturePipeline()

        # Fit on training data
        train_result = pipeline.fit_transform(sample_data.iloc[:80], sample_target.iloc[:80])

        # Transform test data
        test_result = pipeline.transform(sample_data.iloc[80:])

        # Should have same columns
        assert set(train_result.columns) == set(test_result.columns)

    def test_pipeline_produces_model_features(self, sample_data, sample_target):
        """Test pipeline produces expected model features."""
        pipeline = FeaturePipeline()
        result = pipeline.fit_transform(sample_data, sample_target)

        # Check some key features exist
        expected_features = [
            "haversine_distance_km",
            "bearing_sin",
            "hour_of_day",
            "is_weekend",
        ]

        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"

    def test_pipeline_no_nan_in_key_features(self, sample_data, sample_target):
        """Test pipeline doesn't produce NaN in key features."""
        pipeline = FeaturePipeline()
        result = pipeline.fit_transform(sample_data, sample_target)

        # Check key features for NaN
        key_features = ["haversine_distance_km", "bearing_sin", "bearing_cos"]
        for feature in key_features:
            if feature in result.columns:
                assert not result[feature].isna().any(), f"NaN found in {feature}"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_row_transform(self, sample_data, sample_target):
        """Test transformation of single row (inference case)."""
        pipeline = FeaturePipeline()
        pipeline.fit(sample_data, sample_target)

        single_row = sample_data.iloc[[0]]
        result = pipeline.transform(single_row)

        assert len(result) == 1

    def test_missing_optional_columns(self, sample_data, sample_target):
        """Test handling of missing optional columns."""
        # Remove optional column
        data_subset = sample_data.drop(columns=["weather_condition"])

        pipeline = FeaturePipeline()
        # Should not raise error
        result = pipeline.fit_transform(data_subset, sample_target)
        assert len(result) == len(data_subset)

    def test_zero_distance(self):
        """Test handling of zero distance (same pickup/dropoff)."""
        transformer = GeoFeatureTransformer()

        data = pd.DataFrame({
            "pickup_lat": [37.7749],
            "pickup_lng": [-122.4194],
            "dropoff_lat": [37.7749],
            "dropoff_lng": [-122.4194],
        })

        transformer.fit(data)
        result = transformer.transform(data)

        assert result["haversine_distance_km"].iloc[0] == pytest.approx(0, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
