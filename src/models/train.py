"""
Model Training Module

Handles the complete training pipeline:
1. Data loading and preparation
2. Feature engineering
3. Hyperparameter optimization with Optuna
4. Model training with LightGBM (primary) / XGBoost (backup)
5. Cross-validation with time-series awareness
6. Experiment tracking with MLflow
7. Model serialization and registration

Design Philosophy:
- Time-series aware CV to prevent data leakage
- Optimize for MAE (business metric: predict within 5 minutes)
- Log everything for reproducibility
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Note: optuna-integration[lightgbm] has compatibility issues with newer LightGBM
# We use early stopping instead of pruning callbacks
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.features import (
    FeaturePipeline,
    MODEL_FEATURES,
    prepare_features_for_training,
)

warnings.filterwarnings("ignore")
logger.add("logs/training.log", rotation="10 MB")


class DeliveryETATrainer:
    """
    End-to-end trainer for delivery ETA prediction model.

    Supports:
    - LightGBM (primary) and XGBoost (backup)
    - Optuna hyperparameter optimization
    - Time-series cross-validation
    - MLflow experiment tracking
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        mlflow_tracking_uri: str = "sqlite:///models/mlflow.db",
        experiment_name: str = "delivery-eta-prediction",
        random_seed: int = 42,
    ):
        """
        Initialize trainer.

        Args:
            model_type: 'lightgbm' or 'xgboost'
            mlflow_tracking_uri: MLflow tracking server URI
            experiment_name: MLflow experiment name
            random_seed: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_seed = random_seed
        self.model = None
        self.feature_pipeline: Optional[FeaturePipeline] = None
        self.feature_names: List[str] = []
        self.best_params: Dict[str, Any] = {}
        self.metrics: Dict[str, float] = {}

        # Setup MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

        logger.info(f"Trainer initialized: model_type={model_type}, seed={random_seed}")

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate training data."""
        logger.info(f"Loading data from {data_path}")

        df = pd.read_csv(data_path, parse_dates=["order_timestamp"])
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Basic validation
        assert "actual_eta_minutes" in df.columns, "Target column missing"
        assert len(df) > 1000, "Insufficient data for training"

        return df

    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data with time-based train/test split.

        Time-based split ensures no future data leaks into training.
        """
        logger.info("Preparing data with time-based split...")

        # Sort by timestamp
        df = df.sort_values("order_timestamp").reset_index(drop=True)

        # Time-based split
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        logger.info(f"Train: {len(train_df)} samples ({train_df['order_timestamp'].min()} to {train_df['order_timestamp'].max()})")
        logger.info(f"Test: {len(test_df)} samples ({test_df['order_timestamp'].min()} to {test_df['order_timestamp'].max()})")

        # Feature engineering
        y_train = train_df["actual_eta_minutes"]
        y_test = test_df["actual_eta_minutes"]

        # Fit pipeline on training data only
        self.feature_pipeline = FeaturePipeline(
            target_encode_cols=["zone_id", "restaurant_type"],
            agg_cols=["zone_id"],
            add_lag_features=False,
        )

        X_train = self.feature_pipeline.fit_transform(train_df, y_train)
        X_test = self.feature_pipeline.transform(test_df)

        # Select model features
        self.feature_names = [f for f in MODEL_FEATURES if f in X_train.columns]
        X_train = X_train[self.feature_names]
        X_test = X_test[self.feature_names]

        # Fill any remaining NaN
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())

        logger.info(f"Features prepared: {len(self.feature_names)} features")

        return X_train, X_test, y_train, y_test

    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = 100,
        timeout: int = 3600,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Uses time-series CV and optimizes for MAE.
        """
        logger.info(f"Starting Optuna optimization: {n_trials} trials, {timeout}s timeout")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "regression",
                "metric": "mae",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "seed": self.random_seed,
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "num_leaves": trial.suggest_int("num_leaves", 16, 256),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            mae_scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = lgb.LGBMRegressor(**params)

                # Use callbacks for early stopping only (pruning callback has compatibility issues)
                callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=callbacks,
                )

                preds = model.predict(X_val)
                mae = mean_absolute_error(y_val, preds)
                mae_scores.append(mae)

            return np.mean(mae_scores)

        # Create study
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            pruner=optuna.pruners.MedianPruner(),
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        self.best_params = study.best_params
        self.best_params.update({
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "seed": self.random_seed,
        })

        logger.info(f"Best MAE: {study.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")

        return self.best_params

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Train the final model with best parameters.

        Logs everything to MLflow.
        """
        params = params or self.best_params or self._get_default_params()

        logger.info("Training final model...")

        with mlflow.start_run(run_name=f"delivery_eta_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("n_features", len(self.feature_names))
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))

            if self.model_type == "lightgbm":
                self.model = lgb.LGBMRegressor(**params)

                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=100),
                    ],
                )

            else:  # XGBoost backup
                xgb_params = {
                    "objective": "reg:squarederror",
                    "eval_metric": "mae",
                    "n_estimators": params.get("n_estimators", 1000),
                    "learning_rate": params.get("learning_rate", 0.05),
                    "max_depth": params.get("max_depth", 8),
                    "subsample": params.get("subsample", 0.8),
                    "colsample_bytree": params.get("colsample_bytree", 0.8),
                    "random_state": self.random_seed,
                }
                self.model = xgb.XGBRegressor(**xgb_params)
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=100,
                )

            # Evaluate
            self.metrics = self.evaluate(X_test, y_test)

            # Log metrics
            for metric_name, value in self.metrics.items():
                mlflow.log_metric(metric_name, value)

            # Log feature importance
            self._log_feature_importance()

            # Log model
            if self.model_type == "lightgbm":
                mlflow.lightgbm.log_model(
                    self.model,
                    "model",
                    registered_model_name="delivery-eta-model",
                )
            else:
                mlflow.sklearn.log_model(
                    self.model,
                    "model",
                    registered_model_name="delivery-eta-model",
                )

            # Save artifacts
            self._save_artifacts()

            logger.info(f"Training complete. MAE: {self.metrics['mae']:.4f}")

        return self.model

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate model on given data."""
        predictions = self.model.predict(X)

        metrics = {
            "mae": mean_absolute_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
            "mape": mean_absolute_percentage_error(y, predictions) * 100,
            "median_ae": np.median(np.abs(y - predictions)),
        }

        # Custom metrics
        errors = np.abs(y - predictions)
        metrics["within_5min_pct"] = (errors <= 5).mean() * 100
        metrics["within_10min_pct"] = (errors <= 10).mean() * 100
        metrics["p90_error"] = np.percentile(errors, 90)
        metrics["p95_error"] = np.percentile(errors, 95)

        logger.info(f"Evaluation metrics: {metrics}")

        return metrics

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Time-series cross-validation.

        Uses TimeSeriesSplit to ensure temporal ordering.
        """
        logger.info(f"Running {n_splits}-fold time-series cross-validation...")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {
            "mae": [],
            "rmse": [],
            "r2": [],
        }

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**self._get_default_params())
            model.fit(X_tr, y_tr)

            preds = model.predict(X_val)

            cv_results["mae"].append(mean_absolute_error(y_val, preds))
            cv_results["rmse"].append(np.sqrt(mean_squared_error(y_val, preds)))
            cv_results["r2"].append(r2_score(y_val, preds))

            logger.info(f"Fold {fold + 1}: MAE={cv_results['mae'][-1]:.4f}")

        # Log summary
        for metric, values in cv_results.items():
            logger.info(f"CV {metric}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

        return cv_results

    def _get_default_params(self) -> Dict[str, Any]:
        """Get default LightGBM parameters."""
        return {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 8,
            "num_leaves": 64,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbosity": -1,
            "seed": self.random_seed,
            "n_jobs": -1,
        }

    def _log_feature_importance(self):
        """Log feature importance to MLflow."""
        importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

        importance.to_csv("models/feature_importance.csv", index=False)
        mlflow.log_artifact("models/feature_importance.csv")

        logger.info("Top 10 features:")
        for _, row in importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    def _save_artifacts(self):
        """Save model and pipeline artifacts."""
        # Save model
        model_path = "models/delivery_eta_model.joblib"
        joblib.dump(self.model, model_path)
        mlflow.log_artifact(model_path)
        logger.info(f"Model saved to {model_path}")

        # Save feature pipeline
        pipeline_path = "models/feature_pipeline.joblib"
        joblib.dump(self.feature_pipeline, pipeline_path)
        mlflow.log_artifact(pipeline_path)
        logger.info(f"Feature pipeline saved to {pipeline_path}")

        # Save feature names
        feature_path = "models/feature_names.txt"
        with open(feature_path, "w") as f:
            f.write("\n".join(self.feature_names))
        mlflow.log_artifact(feature_path)

        # Save metrics
        metrics_path = "models/metrics.json"
        import json
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)


def main():
    """Main training entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Delivery ETA Model")
    parser.add_argument("--data", type=str, default="data/processed/delivery_orders_clean.csv",
                       help="Path to training data")
    parser.add_argument("--model-type", type=str, default="lightgbm",
                       choices=["lightgbm", "xgboost"])
    parser.add_argument("--optimize", action="store_true",
                       help="Run hyperparameter optimization")
    parser.add_argument("--n-trials", type=int, default=50,
                       help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=1800,
                       help="Optuna timeout in seconds")
    args = parser.parse_args()

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Initialize trainer
    trainer = DeliveryETATrainer(
        model_type=args.model_type,
        mlflow_tracking_uri="sqlite:///models/mlflow.db",
        experiment_name="delivery-eta-prediction",
    )

    # Check if cleaned data exists, if not use raw
    data_path = args.data
    if not os.path.exists(data_path):
        raw_path = "data/raw/delivery_orders.csv"
        if os.path.exists(raw_path):
            logger.info(f"Cleaned data not found, using raw data from {raw_path}")
            data_path = raw_path
        else:
            logger.error("No training data found. Run 'make generate-data' first.")
            return

    # Load data
    df = trainer.load_data(data_path)

    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df, test_size=0.2)

    # Optional: run cross-validation
    cv_results = trainer.cross_validate(X_train, y_train, n_splits=5)

    # Optional: hyperparameter optimization
    if args.optimize:
        best_params = trainer.optimize_hyperparameters(
            X_train, y_train,
            n_trials=args.n_trials,
            timeout=args.timeout,
        )
    else:
        best_params = trainer._get_default_params()

    # Train final model
    model = trainer.train(X_train, y_train, X_test, y_test, params=best_params)

    # Final evaluation
    metrics = trainer.evaluate(X_test, y_test)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: {args.model_type}")
    print(f"Test MAE: {metrics['mae']:.4f} minutes")
    print(f"Test RMSE: {metrics['rmse']:.4f} minutes")
    print(f"Test R²: {metrics['r2']:.4f}")
    print(f"Within 5 min: {metrics['within_5min_pct']:.1f}%")
    print(f"Within 10 min: {metrics['within_10min_pct']:.1f}%")
    print(f"P90 Error: {metrics['p90_error']:.2f} minutes")
    print(f"P95 Error: {metrics['p95_error']:.2f} minutes")

    target_met = metrics['mae'] < 5.0
    print(f"\nTarget MAE < 5 minutes: {'✓ ACHIEVED' if target_met else '✗ NOT MET'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
