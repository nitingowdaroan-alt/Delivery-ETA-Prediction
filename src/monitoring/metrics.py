"""
Prometheus Metrics Module

Provides observability metrics for the delivery ETA prediction service:
1. Request/prediction counts
2. Latency histograms
3. Error rates
4. Model performance metrics
5. Resource utilization

Metrics are exposed via /metrics endpoint for Prometheus scraping.
"""

import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

from loguru import logger
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

# Default registry
registry = REGISTRY


# ============================================================================
# Request/Response Metrics
# ============================================================================

REQUEST_COUNT = Counter(
    "delivery_eta_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "delivery_eta_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

REQUEST_IN_PROGRESS = Gauge(
    "delivery_eta_requests_in_progress",
    "Number of requests currently being processed",
    ["endpoint"],
)


# ============================================================================
# Prediction Metrics
# ============================================================================

PREDICTION_COUNT = Counter(
    "delivery_eta_predictions_total",
    "Total predictions made",
    ["status", "model_version"],
)

PREDICTION_LATENCY = Histogram(
    "delivery_eta_prediction_latency_seconds",
    "Time to generate prediction",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

PREDICTED_ETA = Histogram(
    "delivery_eta_predicted_value_minutes",
    "Distribution of predicted ETA values",
    buckets=[5, 10, 15, 20, 25, 30, 40, 50, 60, 90, 120, 180],
)

PREDICTION_ERROR = Histogram(
    "delivery_eta_prediction_error_minutes",
    "Prediction error when actual is known",
    buckets=[0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30],
)

PREDICTION_ABSOLUTE_ERROR = Summary(
    "delivery_eta_prediction_absolute_error_minutes",
    "Summary of absolute prediction errors",
)


# ============================================================================
# Model Metrics
# ============================================================================

MODEL_INFO = Info(
    "delivery_eta_model",
    "Model metadata",
)

MODEL_LOAD_TIME = Gauge(
    "delivery_eta_model_load_time_seconds",
    "Time taken to load model",
)

MODEL_LAST_PREDICTION = Gauge(
    "delivery_eta_model_last_prediction_timestamp",
    "Timestamp of last prediction",
)

MODEL_FEATURE_COUNT = Gauge(
    "delivery_eta_model_feature_count",
    "Number of features used by model",
)


# ============================================================================
# Data Quality Metrics
# ============================================================================

FEATURE_MISSING_COUNT = Counter(
    "delivery_eta_feature_missing_total",
    "Count of missing feature values",
    ["feature_name"],
)

FEATURE_OUT_OF_RANGE = Counter(
    "delivery_eta_feature_out_of_range_total",
    "Count of out-of-range feature values",
    ["feature_name"],
)

DATA_DRIFT_DETECTED = Gauge(
    "delivery_eta_data_drift_detected",
    "Whether data drift was detected (1=yes, 0=no)",
)

DRIFT_SHARE = Gauge(
    "delivery_eta_drift_share",
    "Proportion of features showing drift",
)


# ============================================================================
# Service Health Metrics
# ============================================================================

SERVICE_UP = Gauge(
    "delivery_eta_service_up",
    "Service health status (1=healthy, 0=unhealthy)",
)

REDIS_CONNECTED = Gauge(
    "delivery_eta_redis_connected",
    "Redis connection status (1=connected, 0=disconnected)",
)

CACHE_HIT_COUNT = Counter(
    "delivery_eta_cache_hits_total",
    "Number of cache hits",
    ["cache_type"],
)

CACHE_MISS_COUNT = Counter(
    "delivery_eta_cache_misses_total",
    "Number of cache misses",
    ["cache_type"],
)


# ============================================================================
# Helper Functions
# ============================================================================

def track_prediction(
    predicted_value: float,
    actual_value: Optional[float] = None,
    latency: float = 0.0,
    model_version: str = "unknown",
    status: str = "success",
):
    """
    Record prediction metrics.

    Args:
        predicted_value: The predicted ETA in minutes
        actual_value: Actual ETA if known (for error calculation)
        latency: Prediction latency in seconds
        model_version: Version of model used
        status: Prediction status (success/error)
    """
    PREDICTION_COUNT.labels(status=status, model_version=model_version).inc()
    PREDICTED_ETA.observe(predicted_value)
    PREDICTION_LATENCY.observe(latency)
    MODEL_LAST_PREDICTION.set(time.time())

    if actual_value is not None:
        error = predicted_value - actual_value
        abs_error = abs(error)
        PREDICTION_ERROR.observe(error)
        PREDICTION_ABSOLUTE_ERROR.observe(abs_error)


def track_request(
    method: str,
    endpoint: str,
    status_code: int,
    latency: float,
):
    """
    Record HTTP request metrics.

    Args:
        method: HTTP method
        endpoint: Request endpoint
        status_code: Response status code
        latency: Request latency in seconds
    """
    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status_code=str(status_code)
    ).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)


def update_model_info(
    model_type: str,
    model_version: str,
    n_features: int,
    load_time: float,
):
    """
    Update model information metrics.

    Args:
        model_type: Type of model (e.g., LGBMRegressor)
        model_version: Model version string
        n_features: Number of features
        load_time: Time taken to load model
    """
    MODEL_INFO.info({
        "model_type": model_type,
        "version": model_version,
    })
    MODEL_FEATURE_COUNT.set(n_features)
    MODEL_LOAD_TIME.set(load_time)


def update_drift_metrics(drift_detected: bool, drift_share: float):
    """
    Update drift detection metrics.

    Args:
        drift_detected: Whether drift was detected
        drift_share: Proportion of drifted features
    """
    DATA_DRIFT_DETECTED.set(1 if drift_detected else 0)
    DRIFT_SHARE.set(drift_share)


def update_health_metrics(service_healthy: bool, redis_connected: bool):
    """
    Update service health metrics.

    Args:
        service_healthy: Overall service health
        redis_connected: Redis connection status
    """
    SERVICE_UP.set(1 if service_healthy else 0)
    REDIS_CONNECTED.set(1 if redis_connected else 0)


def track_cache(cache_type: str, hit: bool):
    """
    Track cache hit/miss.

    Args:
        cache_type: Type of cache (e.g., 'zone_stats', 'restaurant')
        hit: Whether it was a cache hit
    """
    if hit:
        CACHE_HIT_COUNT.labels(cache_type=cache_type).inc()
    else:
        CACHE_MISS_COUNT.labels(cache_type=cache_type).inc()


def track_feature_quality(feature_name: str, is_missing: bool, is_out_of_range: bool):
    """
    Track feature data quality.

    Args:
        feature_name: Name of the feature
        is_missing: Whether value was missing
        is_out_of_range: Whether value was out of expected range
    """
    if is_missing:
        FEATURE_MISSING_COUNT.labels(feature_name=feature_name).inc()
    if is_out_of_range:
        FEATURE_OUT_OF_RANGE.labels(feature_name=feature_name).inc()


# ============================================================================
# Decorators
# ============================================================================

def timed(metric: Histogram):
    """Decorator to time function execution."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                metric.observe(time.time() - start)
        return wrapper
    return decorator


def counted(metric: Counter, labels: Optional[Dict[str, str]] = None):
    """Decorator to count function calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if labels:
                    metric.labels(**labels).inc()
                else:
                    metric.inc()
                return result
            except Exception as e:
                if labels:
                    error_labels = {**labels, "status": "error"}
                    metric.labels(**error_labels).inc()
                raise
        return wrapper
    return decorator


# ============================================================================
# Metrics Collector
# ============================================================================

class MetricsCollector:
    """
    Centralized metrics collection and export.

    Provides a clean interface for recording metrics throughout the application.
    """

    def __init__(self):
        self._start_time = time.time()

    def get_metrics(self) -> bytes:
        """Generate Prometheus metrics output."""
        return generate_latest(registry)

    def get_content_type(self) -> str:
        """Get Prometheus content type."""
        return CONTENT_TYPE_LATEST

    def record_prediction(
        self,
        predicted_eta: float,
        actual_eta: Optional[float] = None,
        latency_seconds: float = 0.0,
        model_version: str = "1.0.0",
        success: bool = True,
    ):
        """Record a prediction with all associated metrics."""
        track_prediction(
            predicted_value=predicted_eta,
            actual_value=actual_eta,
            latency=latency_seconds,
            model_version=model_version,
            status="success" if success else "error",
        )

    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_seconds: float,
    ):
        """Record an HTTP request."""
        track_request(method, path, status_code, duration_seconds)

    def update_service_status(self, healthy: bool, redis_ok: bool):
        """Update service health status."""
        update_health_metrics(healthy, redis_ok)

    def update_model_metrics(
        self,
        model_type: str,
        version: str,
        features: int,
        load_time: float,
    ):
        """Update model-related metrics."""
        update_model_info(model_type, version, features, load_time)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current metrics.

        Returns dict with key metric values.
        """
        return {
            "uptime_seconds": time.time() - self._start_time,
            "service_up": SERVICE_UP._value.get() if hasattr(SERVICE_UP, '_value') else None,
            "redis_connected": REDIS_CONNECTED._value.get() if hasattr(REDIS_CONNECTED, '_value') else None,
        }


# Global collector instance
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector
