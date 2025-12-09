"""
Drift Detection Module

Monitors data and model drift using Evidently AI:
1. Data drift - Feature distribution changes
2. Target drift - Target variable distribution changes
3. Prediction drift - Model output distribution changes
4. Data quality - Missing values, outliers, etc.

Why Evidently?
- Production-ready drift detection
- Beautiful HTML reports
- Integration with MLflow and Prometheus
- Supports both batch and real-time monitoring
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset
from evidently.metrics import (
    ColumnDriftMetric,
    DataDriftTable,
    DatasetDriftMetric,
    RegressionQualityMetric,
)
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset, RegressionTestPreset
from evidently.test_suite import TestSuite
from loguru import logger

logger.add("logs/monitoring.log", rotation="10 MB")


class DriftDetector:
    """
    Detects data and model drift for delivery ETA predictions.

    Provides:
    - Statistical drift detection on features
    - Target drift monitoring
    - Prediction quality degradation alerts
    - HTML reports for visualization
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        target_col: str = "actual_eta_minutes",
        prediction_col: str = "predicted_eta_minutes",
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        drift_threshold: float = 0.1,
    ):
        """
        Initialize drift detector.

        Args:
            reference_data: Historical/training data as reference
            target_col: Name of target column
            prediction_col: Name of prediction column
            numerical_features: List of numerical feature columns
            categorical_features: List of categorical feature columns
            drift_threshold: P-value threshold for drift detection
        """
        self.reference_data = reference_data
        self.target_col = target_col
        self.prediction_col = prediction_col
        self.drift_threshold = drift_threshold

        # Auto-detect feature types if not provided
        if numerical_features is None:
            self.numerical_features = reference_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            # Remove target and prediction columns
            for col in [target_col, prediction_col]:
                if col in self.numerical_features:
                    self.numerical_features.remove(col)
        else:
            self.numerical_features = numerical_features

        if categorical_features is None:
            self.categorical_features = reference_data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        else:
            self.categorical_features = categorical_features

        # Setup column mapping for Evidently
        self.column_mapping = ColumnMapping(
            target=target_col if target_col in reference_data.columns else None,
            prediction=prediction_col if prediction_col in reference_data.columns else None,
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
        )

        logger.info(
            f"DriftDetector initialized: {len(self.numerical_features)} numerical, "
            f"{len(self.categorical_features)} categorical features"
        )

    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect data drift between reference and current data.

        Args:
            current_data: Current production data
            output_path: Path to save HTML report

        Returns:
            (is_drift_detected, drift_metrics)
        """
        logger.info(f"Running data drift detection on {len(current_data)} samples")

        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        # Extract results
        results = report.as_dict()

        drift_detected = results["metrics"][0]["result"]["dataset_drift"]
        drift_share = results["metrics"][0]["result"]["drift_share"]
        drifted_columns = []

        # Get per-column drift
        column_drift = results["metrics"][1]["result"]["drift_by_columns"]
        for col, info in column_drift.items():
            if info.get("drift_detected", False):
                drifted_columns.append({
                    "column": col,
                    "drift_score": info.get("drift_score", 0),
                    "stattest": info.get("stattest_name", "unknown"),
                })

        metrics = {
            "drift_detected": drift_detected,
            "drift_share": drift_share,
            "n_drifted_columns": len(drifted_columns),
            "drifted_columns": drifted_columns,
            "timestamp": datetime.now().isoformat(),
            "n_reference_samples": len(self.reference_data),
            "n_current_samples": len(current_data),
        }

        if output_path:
            report.save_html(output_path)
            logger.info(f"Drift report saved to {output_path}")

        if drift_detected:
            logger.warning(
                f"DATA DRIFT DETECTED: {len(drifted_columns)} columns drifted "
                f"({drift_share*100:.1f}%)"
            )
        else:
            logger.info(f"No significant drift detected (drift_share={drift_share*100:.1f}%)")

        return drift_detected, metrics

    def detect_target_drift(
        self,
        current_data: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect drift in target variable distribution.

        Args:
            current_data: Current data with target variable
            output_path: Path to save HTML report

        Returns:
            (is_drift_detected, drift_metrics)
        """
        if self.target_col not in current_data.columns:
            logger.warning(f"Target column {self.target_col} not in current data")
            return False, {}

        if self.target_col not in self.reference_data.columns:
            logger.warning(f"Target column {self.target_col} not in reference data")
            return False, {}

        logger.info("Running target drift detection")

        report = Report(metrics=[
            ColumnDriftMetric(column_name=self.target_col),
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        results = report.as_dict()
        drift_result = results["metrics"][0]["result"]

        drift_detected = drift_result["drift_detected"]

        metrics = {
            "target_drift_detected": drift_detected,
            "drift_score": drift_result.get("drift_score", 0),
            "stattest": drift_result.get("stattest_name", "unknown"),
            "reference_mean": self.reference_data[self.target_col].mean(),
            "current_mean": current_data[self.target_col].mean(),
            "reference_std": self.reference_data[self.target_col].std(),
            "current_std": current_data[self.target_col].std(),
            "timestamp": datetime.now().isoformat(),
        }

        if output_path:
            report.save_html(output_path)

        if drift_detected:
            logger.warning(f"TARGET DRIFT DETECTED: score={metrics['drift_score']:.4f}")
        else:
            logger.info(f"No target drift detected: score={metrics['drift_score']:.4f}")

        return drift_detected, metrics

    def evaluate_prediction_quality(
        self,
        data: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate prediction quality metrics.

        Args:
            data: Data with actual and predicted values
            output_path: Path to save HTML report

        Returns:
            Quality metrics dictionary
        """
        if self.target_col not in data.columns or self.prediction_col not in data.columns:
            logger.warning("Missing target or prediction columns")
            return {}

        logger.info("Evaluating prediction quality")

        report = Report(metrics=[
            RegressionQualityMetric(),
        ])

        # Create column mapping with prediction
        col_mapping = ColumnMapping(
            target=self.target_col,
            prediction=self.prediction_col,
        )

        report.run(
            reference_data=None,
            current_data=data,
            column_mapping=col_mapping,
        )

        results = report.as_dict()
        quality = results["metrics"][0]["result"]["current"]

        metrics = {
            "mae": quality.get("mean_abs_error", None),
            "rmse": quality.get("rmse", None),
            "mape": quality.get("mean_abs_perc_error", None),
            "r2": quality.get("r2_score", None),
            "me": quality.get("mean_error", None),  # Bias
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(data),
        }

        if output_path:
            report.save_html(output_path)

        logger.info(f"Prediction quality: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

        return metrics

    def run_test_suite(
        self,
        current_data: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Run comprehensive test suite for monitoring.

        Args:
            current_data: Current production data
            output_path: Path to save HTML report

        Returns:
            (all_tests_passed, test_results)
        """
        logger.info("Running monitoring test suite")

        test_suite = TestSuite(tests=[
            DataDriftTestPreset(),
        ])

        test_suite.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        results = test_suite.as_dict()

        # Count passed/failed tests
        n_tests = results["summary"]["total_tests"]
        n_passed = results["summary"]["success_tests"]
        n_failed = results["summary"]["failed_tests"]

        all_passed = n_failed == 0

        metrics = {
            "all_tests_passed": all_passed,
            "n_tests": n_tests,
            "n_passed": n_passed,
            "n_failed": n_failed,
            "pass_rate": n_passed / n_tests if n_tests > 0 else 0,
            "timestamp": datetime.now().isoformat(),
        }

        if output_path:
            test_suite.save_html(output_path)

        if all_passed:
            logger.info(f"All {n_tests} monitoring tests passed")
        else:
            logger.warning(f"{n_failed}/{n_tests} monitoring tests failed")

        return all_passed, metrics

    def generate_full_report(
        self,
        current_data: pd.DataFrame,
        output_dir: str = "reports/monitoring",
    ) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report.

        Args:
            current_data: Current production data
            output_dir: Directory to save reports

        Returns:
            Combined metrics from all checks
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        combined_metrics = {
            "timestamp": datetime.now().isoformat(),
        }

        # Data drift
        drift_detected, drift_metrics = self.detect_data_drift(
            current_data,
            output_path=f"{output_dir}/data_drift_{timestamp}.html"
        )
        combined_metrics["data_drift"] = drift_metrics

        # Target drift (if target available)
        if self.target_col in current_data.columns:
            target_drift, target_metrics = self.detect_target_drift(
                current_data,
                output_path=f"{output_dir}/target_drift_{timestamp}.html"
            )
            combined_metrics["target_drift"] = target_metrics

        # Prediction quality (if predictions available)
        if self.prediction_col in current_data.columns and self.target_col in current_data.columns:
            quality_metrics = self.evaluate_prediction_quality(
                current_data,
                output_path=f"{output_dir}/prediction_quality_{timestamp}.html"
            )
            combined_metrics["prediction_quality"] = quality_metrics

        # Test suite
        all_passed, test_metrics = self.run_test_suite(
            current_data,
            output_path=f"{output_dir}/test_suite_{timestamp}.html"
        )
        combined_metrics["test_suite"] = test_metrics

        # Save combined metrics
        metrics_path = f"{output_dir}/metrics_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(combined_metrics, f, indent=2)

        logger.info(f"Full monitoring report saved to {output_dir}")

        return combined_metrics


def should_retrain(
    drift_metrics: Dict[str, Any],
    quality_metrics: Dict[str, Any],
    mae_threshold: float = 5.0,
    drift_threshold: float = 0.15,
) -> Tuple[bool, List[str]]:
    """
    Determine if model should be retrained based on monitoring metrics.

    Args:
        drift_metrics: Data drift metrics
        quality_metrics: Prediction quality metrics
        mae_threshold: MAE threshold for triggering retrain
        drift_threshold: Drift share threshold for triggering retrain

    Returns:
        (should_retrain, reasons)
    """
    reasons = []

    # Check data drift
    drift_share = drift_metrics.get("drift_share", 0)
    if drift_share > drift_threshold:
        reasons.append(f"High data drift: {drift_share*100:.1f}% > {drift_threshold*100:.1f}%")

    # Check target drift
    if drift_metrics.get("target_drift_detected", False):
        reasons.append("Target drift detected")

    # Check prediction quality
    mae = quality_metrics.get("mae", 0)
    if mae > mae_threshold:
        reasons.append(f"High MAE: {mae:.2f} > {mae_threshold:.2f} minutes")

    should_retrain = len(reasons) > 0

    if should_retrain:
        logger.warning(f"RETRAIN RECOMMENDED: {', '.join(reasons)}")
    else:
        logger.info("No retraining needed based on current metrics")

    return should_retrain, reasons


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, ".")

    # Load reference data
    ref_data = pd.read_csv("data/processed/delivery_orders_clean.csv", nrows=10000)

    # Simulate current data (in production, this would be recent predictions)
    current_data = pd.read_csv("data/processed/delivery_orders_clean.csv", skiprows=10000, nrows=2000)
    current_data.columns = ref_data.columns

    # Initialize detector
    detector = DriftDetector(
        reference_data=ref_data,
        target_col="actual_eta_minutes",
    )

    # Run full monitoring
    metrics = detector.generate_full_report(
        current_data,
        output_dir="reports/monitoring"
    )

    print(json.dumps(metrics, indent=2))
