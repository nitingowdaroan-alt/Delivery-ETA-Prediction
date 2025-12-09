"""
Model Evaluation Module

Comprehensive model evaluation and analysis:
1. Standard regression metrics (MAE, RMSE, R², MAPE)
2. Business-specific metrics (% within 5/10 minutes)
3. Error distribution analysis
4. Feature importance analysis
5. Prediction vs actual plots
6. Residual analysis
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

logger.add("logs/evaluation.log", rotation="10 MB")


class ModelEvaluator:
    """
    Comprehensive model evaluation toolkit.

    Provides:
    - Standard ML metrics
    - Business-specific metrics
    - Visualization tools
    - Segmented analysis
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        output_dir: str = "reports",
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained model with predict method
            feature_names: List of feature names (for importance analysis)
            output_dir: Directory to save reports and plots
        """
        self.model = model
        self.feature_names = feature_names or []
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "test",
    ) -> Dict[str, float]:
        """
        Run full evaluation on dataset.

        Args:
            X: Feature matrix
            y: True labels
            dataset_name: Name for logging/reporting

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model on {dataset_name} set ({len(X)} samples)")

        predictions = self.model.predict(X)
        errors = y - predictions
        abs_errors = np.abs(errors)

        # Standard metrics
        metrics = {
            "mae": mean_absolute_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
            "mape": mean_absolute_percentage_error(y, predictions) * 100,
            "median_ae": np.median(abs_errors),
            "mean_error": np.mean(errors),  # Bias indicator
            "std_error": np.std(errors),
        }

        # Business metrics
        metrics["within_5min_pct"] = (abs_errors <= 5).mean() * 100
        metrics["within_10min_pct"] = (abs_errors <= 10).mean() * 100
        metrics["within_15min_pct"] = (abs_errors <= 15).mean() * 100

        # Error percentiles
        metrics["p50_error"] = np.percentile(abs_errors, 50)
        metrics["p75_error"] = np.percentile(abs_errors, 75)
        metrics["p90_error"] = np.percentile(abs_errors, 90)
        metrics["p95_error"] = np.percentile(abs_errors, 95)
        metrics["p99_error"] = np.percentile(abs_errors, 99)
        metrics["max_error"] = abs_errors.max()

        # Log results
        logger.info(f"Metrics for {dataset_name}:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")

        return metrics

    def segment_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        segment_col: str,
    ) -> pd.DataFrame:
        """
        Analyze performance by segment (e.g., by zone, by hour).

        Args:
            X: Features (must include segment_col)
            y: True labels
            segment_col: Column to segment by

        Returns:
            DataFrame with per-segment metrics
        """
        if segment_col not in X.columns:
            logger.warning(f"Segment column {segment_col} not found")
            return pd.DataFrame()

        predictions = self.model.predict(X)
        errors = np.abs(y - predictions)

        df_analysis = pd.DataFrame({
            "segment": X[segment_col],
            "actual": y,
            "predicted": predictions,
            "abs_error": errors,
        })

        segment_metrics = df_analysis.groupby("segment").agg({
            "abs_error": ["mean", "median", "std", "count"],
            "actual": ["mean", "std"],
        }).reset_index()

        segment_metrics.columns = [
            "segment", "mae", "median_ae", "std_ae", "count",
            "avg_actual_eta", "std_actual_eta"
        ]

        segment_metrics["within_5min_pct"] = df_analysis.groupby("segment").apply(
            lambda g: (g["abs_error"] <= 5).mean() * 100
        ).values

        # Sort by MAE
        segment_metrics = segment_metrics.sort_values("mae")

        return segment_metrics

    def plot_predictions_vs_actual(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Create scatter plot of predictions vs actual values."""
        predictions = self.model.predict(X)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot
        ax = axes[0]
        ax.scatter(y, predictions, alpha=0.3, s=10)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="Perfect prediction")
        ax.set_xlabel("Actual ETA (minutes)")
        ax.set_ylabel("Predicted ETA (minutes)")
        ax.set_title("Predictions vs Actual")
        ax.legend()

        # Add metrics annotation
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        ax.annotate(
            f"MAE: {mae:.2f} min\nR²: {r2:.3f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Residual plot
        ax = axes[1]
        residuals = y - predictions
        ax.scatter(predictions, residuals, alpha=0.3, s=10)
        ax.axhline(y=0, color="r", linestyle="--")
        ax.set_xlabel("Predicted ETA (minutes)")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.set_title("Residual Plot")

        # Add band for ±5 minutes
        ax.axhline(y=5, color="g", linestyle=":", alpha=0.5, label="±5 min")
        ax.axhline(y=-5, color="g", linestyle=":", alpha=0.5)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_error_distribution(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot error distribution histogram and CDF."""
        predictions = self.model.predict(X)
        errors = y - predictions
        abs_errors = np.abs(errors)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Error histogram
        ax = axes[0]
        ax.hist(errors, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="r", linestyle="--", label="Zero error")
        ax.set_xlabel("Error (minutes)")
        ax.set_ylabel("Frequency")
        ax.set_title("Error Distribution")
        ax.legend()

        # Absolute error histogram
        ax = axes[1]
        ax.hist(abs_errors, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(x=5, color="g", linestyle="--", label="5 min threshold")
        ax.set_xlabel("Absolute Error (minutes)")
        ax.set_ylabel("Frequency")
        ax.set_title("Absolute Error Distribution")
        ax.legend()

        # CDF of absolute error
        ax = axes[2]
        sorted_errors = np.sort(abs_errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(sorted_errors, cdf * 100)
        ax.axvline(x=5, color="g", linestyle="--", label="5 min")
        ax.axvline(x=10, color="orange", linestyle="--", label="10 min")
        ax.axhline(y=90, color="gray", linestyle=":", alpha=0.5, label="90%")
        ax.set_xlabel("Absolute Error (minutes)")
        ax.set_ylabel("Cumulative % of Predictions")
        ax.set_title("Error CDF")
        ax.legend()
        ax.set_xlim(0, 30)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_feature_importance(
        self,
        top_n: int = 20,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot feature importance."""
        if not hasattr(self.model, "feature_importances_"):
            logger.warning("Model does not have feature_importances_ attribute")
            return None

        importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=True).tail(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importance["feature"], importance["importance"])
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importance")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_by_segment(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        segment_col: str,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot MAE by segment."""
        segment_metrics = self.segment_analysis(X, y, segment_col)

        if segment_metrics.empty:
            return None

        # Only show segments with reasonable sample size
        segment_metrics = segment_metrics[segment_metrics["count"] >= 100]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # MAE by segment
        ax = axes[0]
        ax.bar(range(len(segment_metrics)), segment_metrics["mae"])
        ax.set_xticks(range(len(segment_metrics)))
        ax.set_xticklabels(segment_metrics["segment"], rotation=45, ha="right")
        ax.set_ylabel("MAE (minutes)")
        ax.set_title(f"MAE by {segment_col}")
        ax.axhline(y=5, color="r", linestyle="--", label="5 min target")
        ax.legend()

        # Within 5min % by segment
        ax = axes[1]
        ax.bar(range(len(segment_metrics)), segment_metrics["within_5min_pct"])
        ax.set_xticks(range(len(segment_metrics)))
        ax.set_xticklabels(segment_metrics["segment"], rotation=45, ha="right")
        ax.set_ylabel("% Within 5 Minutes")
        ax.set_title(f"Accuracy by {segment_col}")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")

        return fig

    def generate_report(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            Report dictionary
        """
        logger.info("Generating evaluation report...")

        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_type": type(self.model).__name__,
            "n_features": len(self.feature_names),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        # Evaluate on both sets
        report["train_metrics"] = self.evaluate(X_train, y_train, "train")
        report["test_metrics"] = self.evaluate(X_test, y_test, "test")

        # Generate plots
        self.plot_predictions_vs_actual(
            X_test, y_test,
            save_path=f"{self.output_dir}/predictions_vs_actual.png"
        )

        self.plot_error_distribution(
            X_test, y_test,
            save_path=f"{self.output_dir}/error_distribution.png"
        )

        self.plot_feature_importance(
            save_path=f"{self.output_dir}/feature_importance.png"
        )

        # Segment analysis
        if "hour_of_day" in X_test.columns:
            hour_metrics = self.segment_analysis(X_test, y_test, "hour_of_day")
            hour_metrics.to_csv(f"{self.output_dir}/metrics_by_hour.csv", index=False)
            report["hour_metrics"] = hour_metrics.to_dict(orient="records")

            self.plot_by_segment(
                X_test, y_test, "hour_of_day",
                save_path=f"{self.output_dir}/metrics_by_hour.png"
            )

        if "day_of_week" in X_test.columns:
            dow_metrics = self.segment_analysis(X_test, y_test, "day_of_week")
            dow_metrics.to_csv(f"{self.output_dir}/metrics_by_dow.csv", index=False)
            report["dow_metrics"] = dow_metrics.to_dict(orient="records")

        # Save report
        report_path = f"{self.output_dir}/evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {report_path}")

        plt.close("all")

        return report


def evaluate_model(
    model_path: str = "models/delivery_eta_model.joblib",
    data_path: str = "data/processed/delivery_orders_clean.csv",
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """
    Main function to evaluate a saved model.

    Args:
        model_path: Path to saved model
        data_path: Path to test data
        output_dir: Directory for outputs

    Returns:
        Evaluation report
    """
    import sys
    sys.path.insert(0, ".")

    from src.data.features import FeaturePipeline, MODEL_FEATURES

    # Load model and pipeline
    model = joblib.load(model_path)
    pipeline = joblib.load("models/feature_pipeline.joblib")

    # Load feature names
    with open("models/feature_names.txt", "r") as f:
        feature_names = [line.strip() for line in f.readlines()]

    # Load data
    df = pd.read_csv(data_path, parse_dates=["order_timestamp"])

    # Time-based split
    df = df.sort_values("order_timestamp")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Transform features
    y_train = train_df["actual_eta_minutes"]
    y_test = test_df["actual_eta_minutes"]

    X_train = pipeline.transform(train_df)[feature_names].fillna(0)
    X_test = pipeline.transform(test_df)[feature_names].fillna(0)

    # Evaluate
    evaluator = ModelEvaluator(model, feature_names, output_dir)
    report = evaluator.generate_report(X_train, y_train, X_test, y_test)

    return report


if __name__ == "__main__":
    evaluate_model()
