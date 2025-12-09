"""
Application Configuration Settings

Centralized configuration management using Pydantic Settings.
Supports environment variables and .env files for secure credential management.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Environment variables can be set directly or via .env file.
    All sensitive credentials (API keys, DB passwords) should be
    provided via environment variables, never hardcoded.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    app_name: str = "Delivery ETA Prediction Service"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"  # development, staging, production

    # Database Configuration (PostgreSQL)
    database_url: str = "postgresql://postgres:postgres@localhost:5432/delivery_eta"
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_seconds: int = 300  # 5 minutes cache TTL

    # Google Maps API
    google_maps_api_key: Optional[str] = None
    google_maps_timeout: int = 10

    # MLflow Configuration
    mlflow_tracking_uri: str = "sqlite:///models/mlflow.db"
    mlflow_experiment_name: str = "delivery-eta-prediction"
    mlflow_model_name: str = "delivery-eta-model"

    # Model Configuration
    model_path: str = "models/delivery_eta_model.joblib"
    model_version: str = "production"  # Model stage in MLflow registry

    # Feature Engineering
    default_traffic_multiplier: float = 1.2
    max_haversine_distance_km: float = 50.0

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = False

    # Monitoring
    prometheus_port: int = 9090
    enable_metrics: bool = True

    # Data Generation (for synthetic data)
    synthetic_data_rows: int = 100000
    random_seed: int = 42

    # Performance thresholds
    target_mae_minutes: float = 5.0
    drift_threshold: float = 0.1

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


class DatabaseSettings(BaseSettings):
    """Database-specific settings."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "localhost"
    port: int = 5432
    name: str = "delivery_eta"
    user: str = "postgres"
    password: str = "postgres"

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class ModelSettings(BaseSettings):
    """Model training and inference settings."""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    # LightGBM hyperparameters (defaults, tuned via Optuna)
    n_estimators: int = 1000
    learning_rate: float = 0.05
    max_depth: int = 8
    num_leaves: int = 64
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1

    # Training configuration
    early_stopping_rounds: int = 50
    cv_folds: int = 5
    test_size: float = 0.2

    # Optuna HPO
    optuna_n_trials: int = 100
    optuna_timeout: int = 3600  # 1 hour


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once
    and reused across the application.
    """
    return Settings()


@lru_cache()
def get_db_settings() -> DatabaseSettings:
    """Get cached database settings."""
    return DatabaseSettings()


@lru_cache()
def get_model_settings() -> ModelSettings:
    """Get cached model settings."""
    return ModelSettings()
