# Delivery ETA Prediction System

A production-grade machine learning system for predicting delivery times, similar to systems used by Uber Eats, DoorDash, and other delivery platforms. This project implements an end-to-end ML pipeline including data ingestion, feature engineering, model training with LightGBM, real-time inference API, and monitoring.

## Features

- **Real-time ETA Prediction**: FastAPI-based async API with sub-100ms latency
- **ML Pipeline**: LightGBM regressor optimized for MAE < 5 minutes
- **Feature Engineering**: Geographic (haversine, bearing), temporal (cyclical encoding), and aggregated features
- **Hyperparameter Optimization**: Optuna integration with time-series aware cross-validation
- **Experiment Tracking**: MLflow for model versioning and experiment management
- **Monitoring**: Evidently for drift detection, Prometheus metrics
- **Production Ready**: Docker deployment, Redis caching, health checks

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Client     │────▶│  FastAPI     │────▶│  ML Model    │
│  (Orders)    │     │  /predict    │     │  (LightGBM)  │
└──────────────┘     └──────────────┘     └──────────────┘
                            │                     │
                            ▼                     ▼
                     ┌──────────────┐     ┌──────────────┐
                     │    Redis     │     │   MLflow     │
                     │   (Cache)    │     │  (Registry)  │
                     └──────────────┘     └──────────────┘
```

## Quick Start

### Prerequisites

- Python 3.9+
- Poetry (recommended) or pip
- Docker & Docker Compose (for full deployment)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd delivery-eta

# Install with Poetry (recommended)
make install-dev

# Or with pip
pip install -e .
```

### Generate Data & Train Model

```bash
# Generate synthetic training data (100k+ orders)
make generate-data

# Train model with default parameters
make train

# Or train with hyperparameter optimization (takes longer)
make train-optimized
```

### Run API Server

```bash
# Development mode with auto-reload
make serve-dev

# Production mode
make serve
```

### Make Predictions

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict/realtime \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_lat": 37.7879,
    "pickup_lng": -122.4074,
    "dropoff_lat": 37.7749,
    "dropoff_lng": -122.4194,
    "zone_id": 1,
    "restaurant_type": "casual",
    "traffic_multiplier": 1.2,
    "weather_condition": "clear",
    "prep_time_minutes": 15
  }'
```

**Sample Response:**
```json
{
  "predicted_eta_minutes": 28.5,
  "confidence_interval_lower": 21.8,
  "confidence_interval_upper": 35.3,
  "model_version": "1.0.0",
  "prediction_id": "a1b2c3d4e5f6",
  "metadata": {
    "latency_ms": 12.5,
    "features_used": 28
  }
}
```

## Docker Deployment

```bash
# Build and start all services
make docker-up

# Services available:
# - API:        http://localhost:8000
# - MLflow:     http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana:    http://localhost:3000

# View logs
make docker-logs

# Stop services
make docker-down
```

## Project Structure

```
delivery-eta/
├── src/
│   ├── data/
│   │   ├── generate.py      # Synthetic data generation
│   │   ├── ingest.py        # Data loading (DB, CSV, Google Maps)
│   │   ├── clean.py         # Data cleaning and validation
│   │   └── features.py      # Feature engineering pipeline
│   ├── models/
│   │   ├── train.py         # Model training with MLflow
│   │   └── evaluate.py      # Model evaluation and reporting
│   ├── api/
│   │   └── main.py          # FastAPI application
│   └── monitoring/
│       ├── drift.py         # Evidently drift detection
│       └── metrics.py       # Prometheus metrics
├── tests/
│   ├── test_features.py     # Feature engineering tests
│   ├── test_api.py          # API integration tests
│   └── load_test.py         # Locust load testing
├── config/
│   ├── settings.py          # Application configuration
│   └── prometheus.yml       # Prometheus config
├── models/                   # Saved models and artifacts
├── data/
│   ├── raw/                  # Raw data files
│   └── processed/            # Processed data files
├── docker-compose.yml        # Docker services
├── Dockerfile                # API container
├── pyproject.toml            # Poetry dependencies
├── Makefile                  # Commands
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and service status |
| `/predict/realtime` | POST | Real-time ETA prediction |
| `/predict/batch` | POST | Batch predictions (up to 1000) |
| `/model/info` | GET | Model metadata and metrics |
| `/metrics` | GET | Prometheus metrics |

### Request Schema

```python
{
    "pickup_lat": float,      # Required: -90 to 90
    "pickup_lng": float,      # Required: -180 to 180
    "dropoff_lat": float,     # Required: -90 to 90
    "dropoff_lng": float,     # Required: -180 to 180
    "restaurant_id": str,     # Optional
    "zone_id": int,           # Optional: 1-100
    "restaurant_type": str,   # Optional: fast_food, casual, etc.
    "order_time": datetime,   # Optional: defaults to now
    "traffic_multiplier": float,  # Optional: 0.5-3.0
    "weather_condition": str, # Optional: clear, rain, etc.
    "prep_time_minutes": float,   # Optional: 1-60
    "rider_avg_speed_kmh": float  # Optional: 5-50
}
```

## Model Details

### Features Used

| Category | Features |
|----------|----------|
| Geographic | haversine_distance, bearing_sin, bearing_cos, lat_diff, lng_diff |
| Temporal | hour_of_day, day_of_week, hour_sin, hour_cos, is_rush_hour |
| Categorical | zone_id_encoded, restaurant_type_encoded |
| Aggregated | zone_eta_mean, zone_eta_std, zone_order_count |
| External | traffic_multiplier, weather_multiplier, prep_time_minutes |

### Training Configuration

- **Algorithm**: LightGBM (Gradient Boosting)
- **Target**: `actual_eta_minutes` (regression)
- **Optimization**: Optuna TPE sampler
- **CV Strategy**: TimeSeriesSplit (5 folds, no data leakage)
- **Metric**: MAE (Mean Absolute Error)

### Expected Performance

| Metric | Target | Typical |
|--------|--------|---------|
| MAE | < 5 min | 3.5-4.5 min |
| RMSE | < 7 min | 5-6 min |
| R² | > 0.7 | 0.75-0.85 |
| Within 5 min | > 60% | 65-75% |
| P90 Error | < 10 min | 7-9 min |

## Testing

```bash
# Run all tests
make test

# Unit tests only
make test-unit

# Integration tests
make test-integration

# Load tests (requires running server)
make serve &
make test-load
```

## Monitoring

### Drift Detection

```bash
# Run drift detection
make monitor-drift

# Reports generated in reports/monitoring/
```

### Prometheus Metrics

Available metrics:
- `delivery_eta_predictions_total` - Prediction count by status
- `delivery_eta_prediction_latency_seconds` - Latency histogram
- `delivery_eta_predicted_value_minutes` - Prediction distribution
- `delivery_eta_data_drift_detected` - Drift indicator

### Grafana Dashboards

Access Grafana at http://localhost:3000 (admin/admin) when using Docker.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection | sqlite |
| `REDIS_URL` | Redis connection | localhost:6379 |
| `MLFLOW_TRACKING_URI` | MLflow server | sqlite:///models/mlflow.db |
| `GOOGLE_MAPS_API_KEY` | Google Maps API key | None |
| `API_PORT` | API server port | 8000 |
| `MODEL_PATH` | Path to model file | models/delivery_eta_model.joblib |

## Development

### Code Style

```bash
# Format code
make format

# Check linting
make lint
```

### Adding Features

1. Add feature engineering in `src/data/features.py`
2. Update `MODEL_FEATURES` list
3. Add tests in `tests/test_features.py`
4. Retrain model: `make train`

### Retraining Workflow

```bash
# Full pipeline: generate data, clean, train, evaluate
make pipeline

# View results in MLflow
make mlflow-ui
```

## Production Deployment

### AWS/GCP Deployment

1. Build and push Docker image:
```bash
docker build -t delivery-eta-api:latest .
docker push <registry>/delivery-eta-api:latest
```

2. Deploy with Kubernetes or ECS:
- Set environment variables
- Configure Redis/PostgreSQL endpoints
- Set up load balancer for `/health` checks
- Enable Prometheus metrics scraping

### Scaling Recommendations

- **API**: 4+ workers, 2GB RAM per instance
- **Redis**: 256MB+ for zone stats cache
- **Model Loading**: Use model registry for versioning

## Troubleshooting

### Common Issues

**Model not loading:**
```bash
# Ensure model file exists
ls -la models/delivery_eta_model.joblib

# Retrain if needed
make train
```

**Redis connection failed:**
```bash
# Start Redis with Docker
docker run -d -p 6379:6379 redis:alpine
```

**High latency:**
- Check Redis cache hit rate
- Verify feature pipeline efficiency
- Consider batch predictions for bulk requests

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## Acknowledgments

- [LightGBM](https://lightgbm.readthedocs.io/) - Gradient boosting framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [Evidently](https://evidentlyai.com/) - ML monitoring
