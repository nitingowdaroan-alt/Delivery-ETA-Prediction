"""
Integration Tests for FastAPI Application

Tests:
1. Health check endpoint
2. Prediction endpoint
3. Batch prediction endpoint
4. Error handling
5. Input validation
"""

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, ".")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.predict.return_value = np.array([25.5])
    model.feature_importances_ = np.random.rand(20)
    return model


@pytest.fixture
def mock_pipeline():
    """Create a mock feature pipeline."""
    import pandas as pd

    pipeline = MagicMock()

    def mock_transform(df):
        # Return DataFrame with expected features
        return pd.DataFrame({
            "haversine_distance_km": [2.5],
            "bearing_sin": [0.5],
            "bearing_cos": [0.5],
            "lat_diff": [0.01],
            "lng_diff": [0.01],
            "manhattan_proxy": [0.02],
            "hour_of_day": [12],
            "day_of_week": [1],
            "hour_sin": [0.5],
            "hour_cos": [0.5],
            "dow_sin": [0.5],
            "dow_cos": [0.5],
            "is_morning_rush": [0],
            "is_lunch_rush": [1],
            "is_dinner_rush": [0],
            "is_late_night": [0],
            "is_weekend": [0],
            "zone_id_encoded": [25.0],
            "restaurant_type_encoded": [20.0],
            "zone_id_eta_mean": [25.0],
            "zone_id_eta_std": [8.0],
            "zone_id_order_count": [1000],
            "weather_encoded": [0],
            "weather_multiplier": [1.0],
            "traffic_multiplier": [1.2],
            "rider_avg_speed_kmh": [25.0],
            "rider_reliability": [0.95],
            "prep_time_minutes": [15.0],
        })

    pipeline.transform.side_effect = mock_transform
    return pipeline


@pytest.fixture
def client(mock_model, mock_pipeline):
    """Create test client with mocked dependencies."""
    from src.api.main import app, app_state

    # Set up mock state
    app_state.model = mock_model
    app_state.feature_pipeline = mock_pipeline
    app_state.feature_names = [
        "haversine_distance_km", "bearing_sin", "bearing_cos",
        "lat_diff", "lng_diff", "manhattan_proxy",
        "hour_of_day", "day_of_week", "hour_sin", "hour_cos",
        "dow_sin", "dow_cos", "is_morning_rush", "is_lunch_rush",
        "is_dinner_rush", "is_late_night", "is_weekend",
        "zone_id_encoded", "restaurant_type_encoded",
        "zone_id_eta_mean", "zone_id_eta_std", "zone_id_order_count",
        "weather_encoded", "weather_multiplier", "traffic_multiplier",
        "rider_avg_speed_kmh", "rider_reliability", "prep_time_minutes",
    ]
    app_state.model_version = "1.0.0-test"
    app_state.training_metrics = {"mae": 4.5, "rmse": 6.2}

    return TestClient(app)


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check_healthy(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "uptime_seconds" in data

    def test_health_check_model_not_loaded(self, mock_pipeline):
        """Test health check when model is not loaded."""
        from src.api.main import app, app_state

        app_state.model = None
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


# =============================================================================
# Prediction Endpoint Tests
# =============================================================================

class TestPredictionEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_success(self, client):
        """Test successful prediction."""
        payload = {
            "pickup_lat": 37.7879,
            "pickup_lng": -122.4074,
            "dropoff_lat": 37.7749,
            "dropoff_lng": -122.4194,
            "restaurant_id": "R0001",
            "zone_id": 1,
            "restaurant_type": "casual",
            "traffic_multiplier": 1.2,
            "weather_condition": "clear",
            "prep_time_minutes": 15,
        }

        response = client.post("/predict/realtime", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "predicted_eta_minutes" in data
        assert "confidence_interval_lower" in data
        assert "confidence_interval_upper" in data
        assert "prediction_id" in data
        assert data["model_version"] == "1.0.0-test"

    def test_predict_minimal_payload(self, client):
        """Test prediction with minimal required fields."""
        payload = {
            "pickup_lat": 37.7879,
            "pickup_lng": -122.4074,
            "dropoff_lat": 37.7749,
            "dropoff_lng": -122.4194,
        }

        response = client.post("/predict/realtime", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "predicted_eta_minutes" in data

    def test_predict_invalid_latitude(self, client):
        """Test prediction with invalid latitude."""
        payload = {
            "pickup_lat": 100.0,  # Invalid: > 90
            "pickup_lng": -122.4074,
            "dropoff_lat": 37.7749,
            "dropoff_lng": -122.4194,
        }

        response = client.post("/predict/realtime", json=payload)

        assert response.status_code == 422  # Validation error

    def test_predict_invalid_longitude(self, client):
        """Test prediction with invalid longitude."""
        payload = {
            "pickup_lat": 37.7879,
            "pickup_lng": -200.0,  # Invalid: < -180
            "dropoff_lat": 37.7749,
            "dropoff_lng": -122.4194,
        }

        response = client.post("/predict/realtime", json=payload)

        assert response.status_code == 422

    def test_predict_with_order_time(self, client):
        """Test prediction with explicit order time."""
        payload = {
            "pickup_lat": 37.7879,
            "pickup_lng": -122.4074,
            "dropoff_lat": 37.7749,
            "dropoff_lng": -122.4194,
            "order_time": "2023-06-15T18:30:00",
        }

        response = client.post("/predict/realtime", json=payload)

        assert response.status_code == 200

    def test_predict_model_not_loaded(self, mock_pipeline):
        """Test prediction when model is not loaded."""
        from src.api.main import app, app_state

        app_state.model = None
        client = TestClient(app)

        payload = {
            "pickup_lat": 37.7879,
            "pickup_lng": -122.4074,
            "dropoff_lat": 37.7749,
            "dropoff_lng": -122.4194,
        }

        response = client.post("/predict/realtime", json=payload)

        assert response.status_code == 503  # Service unavailable


# =============================================================================
# Batch Prediction Tests
# =============================================================================

class TestBatchPrediction:
    """Tests for batch prediction endpoint."""

    def test_batch_predict_success(self, client):
        """Test successful batch prediction."""
        payload = {
            "orders": [
                {
                    "pickup_lat": 37.7879,
                    "pickup_lng": -122.4074,
                    "dropoff_lat": 37.7749,
                    "dropoff_lng": -122.4194,
                },
                {
                    "pickup_lat": 37.7899,
                    "pickup_lng": -122.4094,
                    "dropoff_lat": 37.7769,
                    "dropoff_lng": -122.4214,
                },
            ]
        }

        response = client.post("/predict/batch", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2
        assert len(data["predictions"]) == 2
        assert "processing_time_ms" in data

    def test_batch_predict_empty_list(self, client):
        """Test batch prediction with empty list."""
        payload = {"orders": []}

        response = client.post("/predict/batch", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0

    def test_batch_predict_single_item(self, client):
        """Test batch prediction with single item."""
        payload = {
            "orders": [
                {
                    "pickup_lat": 37.7879,
                    "pickup_lng": -122.4074,
                    "dropoff_lat": 37.7749,
                    "dropoff_lng": -122.4194,
                }
            ]
        }

        response = client.post("/predict/batch", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1


# =============================================================================
# Model Info Endpoint Tests
# =============================================================================

class TestModelInfo:
    """Tests for model info endpoint."""

    def test_model_info_success(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")

        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "model_version" in data
        assert "n_features" in data
        assert "feature_names" in data
        assert "training_metrics" in data


# =============================================================================
# Metrics Endpoint Tests
# =============================================================================

class TestMetricsEndpoint:
    """Tests for Prometheus metrics endpoint."""

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"] or \
               "text/plain" in str(response.headers.get("content-type", ""))


# =============================================================================
# Input Validation Tests
# =============================================================================

class TestInputValidation:
    """Tests for request input validation."""

    def test_missing_required_field(self, client):
        """Test request with missing required field."""
        payload = {
            "pickup_lat": 37.7879,
            # Missing pickup_lng
            "dropoff_lat": 37.7749,
            "dropoff_lng": -122.4194,
        }

        response = client.post("/predict/realtime", json=payload)

        assert response.status_code == 422

    def test_invalid_traffic_multiplier(self, client):
        """Test invalid traffic multiplier range."""
        payload = {
            "pickup_lat": 37.7879,
            "pickup_lng": -122.4074,
            "dropoff_lat": 37.7749,
            "dropoff_lng": -122.4194,
            "traffic_multiplier": 5.0,  # Invalid: > 3.0
        }

        response = client.post("/predict/realtime", json=payload)

        assert response.status_code == 422

    def test_invalid_json(self, client):
        """Test request with invalid JSON."""
        response = client.post(
            "/predict/realtime",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
