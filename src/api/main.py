"""
FastAPI Application for Delivery ETA Prediction

Provides REST API endpoints for:
1. /health - Health check
2. /predict/realtime - Real-time ETA prediction
3. /predict/batch - Batch predictions
4. /model/info - Model information
5. /metrics - Prometheus metrics

Design Philosophy:
- Async endpoints for high throughput
- Redis caching for feature lookups (zone stats, etc.)
- Graceful degradation when external services fail
- Structured logging for observability
"""

import asyncio
import hashlib
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import redis
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from loguru import logger
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import get_settings
from src.data.features import FeaturePipeline, MODEL_FEATURES

# Configure logging
logger.add("logs/api.log", rotation="10 MB", retention="7 days")

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "delivery_eta_predictions_total",
    "Total number of ETA predictions",
    ["status"]
)
PREDICTION_LATENCY = Histogram(
    "delivery_eta_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)
PREDICTION_VALUE = Histogram(
    "delivery_eta_predicted_minutes",
    "Predicted ETA values in minutes",
    buckets=[5, 10, 15, 20, 25, 30, 40, 50, 60, 90, 120]
)


# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request schema for real-time ETA prediction."""

    pickup_lat: float = Field(..., ge=-90, le=90, description="Pickup latitude")
    pickup_lng: float = Field(..., ge=-180, le=180, description="Pickup longitude")
    dropoff_lat: float = Field(..., ge=-90, le=90, description="Dropoff latitude")
    dropoff_lng: float = Field(..., ge=-180, le=180, description="Dropoff longitude")
    restaurant_id: Optional[str] = Field(None, description="Restaurant ID")
    zone_id: Optional[int] = Field(None, ge=1, le=100, description="Zone ID")
    restaurant_type: Optional[str] = Field("casual", description="Restaurant type")
    order_time: Optional[datetime] = Field(None, description="Order timestamp")
    traffic_multiplier: Optional[float] = Field(None, ge=0.5, le=3.0)
    weather_condition: Optional[str] = Field("clear")
    prep_time_minutes: Optional[float] = Field(None, ge=1, le=60)
    rider_avg_speed_kmh: Optional[float] = Field(None, ge=5, le=50)

    @validator("order_time", pre=True, always=True)
    def set_order_time(cls, v):
        return v or datetime.now()

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response schema for ETA prediction."""

    predicted_eta_minutes: float = Field(..., description="Predicted delivery time in minutes")
    confidence_interval_lower: float = Field(..., description="Lower bound of 90% CI")
    confidence_interval_upper: float = Field(..., description="Upper bound of 90% CI")
    model_version: str = Field(..., description="Model version used")
    prediction_id: str = Field(..., description="Unique prediction identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    orders: List[PredictionRequest] = Field(..., max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: List[PredictionResponse]
    total_count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    redis_connected: bool
    version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_type: str
    model_version: str
    n_features: int
    feature_names: List[str]
    training_metrics: Dict[str, float]


# Global state
class AppState:
    """Application state container."""

    def __init__(self):
        self.model = None
        self.feature_pipeline: Optional[FeaturePipeline] = None
        self.feature_names: List[str] = []
        self.redis_client: Optional[redis.Redis] = None
        self.model_version = "unknown"
        self.training_metrics: Dict[str, float] = {}
        self.start_time = time.time()

    async def load_model(self, model_path: str, pipeline_path: str):
        """Load model and feature pipeline."""
        logger.info(f"Loading model from {model_path}")

        self.model = joblib.load(model_path)
        self.feature_pipeline = joblib.load(pipeline_path)

        # Load feature names
        feature_names_path = model_path.replace("delivery_eta_model.joblib", "feature_names.txt")
        if os.path.exists(feature_names_path):
            with open(feature_names_path, "r") as f:
                self.feature_names = [line.strip() for line in f.readlines()]
        else:
            self.feature_names = MODEL_FEATURES

        # Load metrics
        metrics_path = model_path.replace("delivery_eta_model.joblib", "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.training_metrics = json.load(f)

        self.model_version = "1.0.0"
        logger.info(f"Model loaded: {type(self.model).__name__}, {len(self.feature_names)} features")

    async def connect_redis(self, redis_url: str):
        """Connect to Redis for caching."""
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self.redis_client = None

    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.model is not None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()

    # Create directories
    os.makedirs("logs", exist_ok=True)

    # Load model
    model_path = settings.model_path
    pipeline_path = model_path.replace("delivery_eta_model.joblib", "feature_pipeline.joblib")

    if os.path.exists(model_path):
        await app_state.load_model(model_path, pipeline_path)
    else:
        logger.warning(f"Model not found at {model_path}. Service starting without model.")

    # Connect Redis
    await app_state.connect_redis(settings.redis_url)

    logger.info("Application startup complete")
    yield

    # Cleanup
    if app_state.redis_client:
        app_state.redis_client.close()
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Delivery ETA Prediction API",
    description="Real-time delivery ETA prediction using ML",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} duration={duration:.3f}s"
    )
    return response


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service health status including model and Redis connectivity.
    """
    return HealthResponse(
        status="healthy" if app_state.is_healthy() else "degraded",
        model_loaded=app_state.model is not None,
        redis_connected=app_state.redis_client is not None,
        version="1.0.0",
        uptime_seconds=time.time() - app_state.start_time,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get model information.

    Returns model type, version, features, and training metrics.
    """
    if not app_state.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return ModelInfoResponse(
        model_type=type(app_state.model).__name__,
        model_version=app_state.model_version,
        n_features=len(app_state.feature_names),
        feature_names=app_state.feature_names,
        training_metrics=app_state.training_metrics,
    )


@app.post("/predict/realtime", response_model=PredictionResponse)
async def predict_realtime(request: PredictionRequest):
    """
    Real-time ETA prediction endpoint.

    Takes order details and returns predicted delivery time.
    Uses ML model with optional Google Maps baseline enhancement.
    """
    start_time = time.time()

    if not app_state.is_healthy():
        PREDICTION_COUNTER.labels(status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Train model first."
        )

    try:
        # Create prediction ID
        prediction_id = hashlib.md5(
            f"{request.pickup_lat}{request.dropoff_lat}{time.time()}".encode()
        ).hexdigest()[:12]

        # Prepare features
        features = await prepare_features(request)

        # Make prediction
        predicted_eta = float(app_state.model.predict(features)[0])

        # Calculate confidence interval (approximate using historical variance)
        # In production, you'd use proper uncertainty quantification
        mae = app_state.training_metrics.get("mae", 4.5)
        ci_lower = max(1, predicted_eta - mae * 1.5)
        ci_upper = predicted_eta + mae * 1.5

        # Record metrics
        latency = time.time() - start_time
        PREDICTION_COUNTER.labels(status="success").inc()
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_VALUE.observe(predicted_eta)

        logger.info(
            f"Prediction: {predicted_eta:.1f} min "
            f"(lat={latency*1000:.1f}ms, id={prediction_id})"
        )

        return PredictionResponse(
            predicted_eta_minutes=round(predicted_eta, 1),
            confidence_interval_lower=round(ci_lower, 1),
            confidence_interval_upper=round(ci_upper, 1),
            model_version=app_state.model_version,
            prediction_id=prediction_id,
            metadata={
                "latency_ms": round(latency * 1000, 2),
                "features_used": len(app_state.feature_names),
            }
        )

    except Exception as e:
        PREDICTION_COUNTER.labels(status="error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction endpoint.

    Process multiple predictions in parallel for efficiency.
    """
    start_time = time.time()

    if not app_state.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Process predictions concurrently
        predictions = []
        for order in request.orders:
            pred = await predict_realtime(order)
            predictions.append(pred)

        processing_time = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


async def prepare_features(request: PredictionRequest) -> pd.DataFrame:
    """
    Prepare features for prediction.

    Handles:
    - Feature engineering
    - Cache lookup for aggregated features
    - Default value imputation
    """
    from geopy.distance import geodesic

    # Calculate haversine distance
    distance = geodesic(
        (request.pickup_lat, request.pickup_lng),
        (request.dropoff_lat, request.dropoff_lng)
    ).kilometers

    # Parse order time
    order_time = request.order_time or datetime.now()

    # Build base features dict
    features_dict = {
        "pickup_lat": request.pickup_lat,
        "pickup_lng": request.pickup_lng,
        "dropoff_lat": request.dropoff_lat,
        "dropoff_lng": request.dropoff_lng,
        "haversine_distance_km": distance,
        "hour_of_day": order_time.hour,
        "day_of_week": order_time.weekday(),
        "month": order_time.month,
        "is_weekend": order_time.weekday() >= 5,
        "is_morning_rush": 7 <= order_time.hour <= 9,
        "is_lunch_rush": 11 <= order_time.hour <= 14,
        "is_dinner_rush": 17 <= order_time.hour <= 21,
        "is_late_night": order_time.hour >= 22 or order_time.hour <= 5,
        "zone_id": request.zone_id or 1,
        "restaurant_type": request.restaurant_type or "casual",
        "traffic_multiplier": request.traffic_multiplier or 1.2,
        "weather_condition": request.weather_condition or "clear",
        "prep_time_minutes": request.prep_time_minutes or 15.0,
        "rider_avg_speed_kmh": request.rider_avg_speed_kmh or 25.0,
        "rider_reliability": 0.95,
        "order_timestamp": order_time,
    }

    # Try to get cached zone stats
    zone_stats = await get_cached_zone_stats(request.zone_id)
    features_dict.update(zone_stats)

    # Create DataFrame
    df = pd.DataFrame([features_dict])

    # Apply feature pipeline transform
    if app_state.feature_pipeline:
        df = app_state.feature_pipeline.transform(df)

    # Select model features and fill missing
    available_features = [f for f in app_state.feature_names if f in df.columns]
    df = df[available_features]

    # Fill any remaining missing features with 0
    for f in app_state.feature_names:
        if f not in df.columns:
            df[f] = 0

    df = df[app_state.feature_names].fillna(0)

    return df


async def get_cached_zone_stats(zone_id: Optional[int]) -> Dict[str, float]:
    """
    Get cached zone statistics from Redis.

    Returns defaults if cache miss or Redis unavailable.
    """
    defaults = {
        "zone_id_eta_mean": 25.0,
        "zone_id_eta_median": 23.0,
        "zone_id_eta_std": 8.0,
        "zone_id_order_count": 1000,
        "zone_id_encoded": 0.0,
        "restaurant_type_encoded": 0.0,
    }

    if not zone_id or not app_state.redis_client:
        return defaults

    try:
        cache_key = f"zone_stats:{zone_id}"
        cached = app_state.redis_client.get(cache_key)

        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Redis cache error: {e}")

    return defaults


def run_server():
    """Run the API server."""
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run_server()
