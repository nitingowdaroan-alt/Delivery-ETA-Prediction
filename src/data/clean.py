"""
Data Cleaning Module

Handles data validation, cleaning, and preprocessing:
1. Missing value imputation
2. Outlier detection and treatment
3. Data type validation
4. Consistency checks

Design Philosophy:
- Make cleaning operations idempotent (can run multiple times safely)
- Log all transformations for reproducibility
- Preserve original data statistics for comparison
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder

logger.add("logs/cleaning.log", rotation="10 MB")


class DataCleaner:
    """
    Comprehensive data cleaning pipeline for delivery order data.

    Handles common data quality issues:
    - Missing values (various imputation strategies)
    - Outliers (IQR and domain-specific bounds)
    - Invalid coordinates
    - Duplicate records
    - Data type mismatches
    """

    # Expected columns and their types
    EXPECTED_COLUMNS = {
        "order_id": "string",
        "order_timestamp": "datetime",
        "restaurant_id": "string",
        "zone_id": "int",
        "restaurant_type": "string",
        "pickup_lat": "float",
        "pickup_lng": "float",
        "dropoff_lat": "float",
        "dropoff_lng": "float",
        "haversine_distance_km": "float",
        "hour_of_day": "int",
        "day_of_week": "int",
        "is_weekend": "bool",
        "traffic_multiplier": "float",
        "weather_condition": "string",
        "actual_eta_minutes": "float",
    }

    # Valid ranges for numeric columns
    VALID_RANGES = {
        "pickup_lat": (-90, 90),
        "pickup_lng": (-180, 180),
        "dropoff_lat": (-90, 90),
        "dropoff_lng": (-180, 180),
        "haversine_distance_km": (0, 100),  # Max 100km delivery seems reasonable
        "hour_of_day": (0, 23),
        "day_of_week": (0, 6),
        "traffic_multiplier": (0.5, 3.0),
        "actual_eta_minutes": (1, 180),  # 1 min to 3 hours
    }

    def __init__(
        self,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        fill_strategy: str = "median",
    ):
        """
        Initialize data cleaner.

        Args:
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'bounds')
            outlier_threshold: Threshold for outlier detection
            fill_strategy: Strategy for filling missing values ('median', 'mean', 'mode')
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.fill_strategy = fill_strategy
        self.cleaning_stats: Dict[str, Any] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run full cleaning pipeline.

        Args:
            df: Raw dataframe

        Returns:
            Cleaned dataframe
        """
        logger.info(f"Starting cleaning pipeline. Initial shape: {df.shape}")
        original_len = len(df)

        # Store original stats
        self.cleaning_stats["original_rows"] = original_len
        self.cleaning_stats["original_columns"] = len(df.columns)

        # Step 1: Remove duplicates
        df = self._remove_duplicates(df)

        # Step 2: Validate and fix data types
        df = self._validate_dtypes(df)

        # Step 3: Handle missing values
        df = self._handle_missing(df)

        # Step 4: Validate ranges and remove invalid rows
        df = self._validate_ranges(df)

        # Step 5: Handle outliers in target variable
        df = self._handle_outliers(df, "actual_eta_minutes")

        # Final stats
        self.cleaning_stats["final_rows"] = len(df)
        self.cleaning_stats["rows_removed"] = original_len - len(df)
        self.cleaning_stats["removal_pct"] = (original_len - len(df)) / original_len * 100

        logger.info(f"Cleaning complete. Final shape: {df.shape}")
        logger.info(f"Removed {self.cleaning_stats['rows_removed']} rows "
                   f"({self.cleaning_stats['removal_pct']:.2f}%)")

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows based on order_id."""
        if "order_id" in df.columns:
            n_dupes = df.duplicated(subset=["order_id"]).sum()
            if n_dupes > 0:
                df = df.drop_duplicates(subset=["order_id"], keep="first")
                logger.info(f"Removed {n_dupes} duplicate order_ids")
                self.cleaning_stats["duplicates_removed"] = n_dupes
        return df

    def _validate_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types."""
        # Convert timestamp
        if "order_timestamp" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["order_timestamp"]):
                df["order_timestamp"] = pd.to_datetime(df["order_timestamp"], errors="coerce")

        # Convert boolean columns
        bool_cols = ["is_weekend", "is_lunch_rush", "is_dinner_rush"]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        # Convert integer columns
        int_cols = ["zone_id", "hour_of_day", "day_of_week", "rider_completed_deliveries"]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        # Convert float columns
        float_cols = [
            "pickup_lat", "pickup_lng", "dropoff_lat", "dropoff_lng",
            "haversine_distance_km", "traffic_multiplier", "weather_multiplier",
            "prep_time_minutes", "actual_eta_minutes", "rider_avg_speed_kmh",
            "rider_reliability"
        ]
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies."""
        missing_before = df.isnull().sum().sum()

        # Categorical columns: fill with mode or 'unknown'
        cat_cols = ["restaurant_id", "restaurant_type", "weather_condition", "rider_id"]
        for col in cat_cols:
            if col in df.columns:
                if df[col].isnull().any():
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val.iloc[0])
                    else:
                        df[col] = df[col].fillna("unknown")

        # Numeric columns: fill with median or mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                if self.fill_strategy == "median":
                    fill_val = df[col].median()
                elif self.fill_strategy == "mean":
                    fill_val = df[col].mean()
                else:
                    fill_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 0

                df[col] = df[col].fillna(fill_val)
                logger.debug(f"Filled {col} missing values with {fill_val:.2f}")

        # Boolean columns: fill with False
        bool_cols = df.select_dtypes(include=["bool"]).columns
        for col in bool_cols:
            df[col] = df[col].fillna(False)

        missing_after = df.isnull().sum().sum()
        self.cleaning_stats["missing_values_filled"] = missing_before - missing_after

        if missing_before > 0:
            logger.info(f"Filled {missing_before - missing_after} missing values")

        return df

    def _validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate numeric columns are within expected ranges."""
        rows_before = len(df)

        for col, (min_val, max_val) in self.VALID_RANGES.items():
            if col in df.columns:
                invalid_mask = (df[col] < min_val) | (df[col] > max_val)
                n_invalid = invalid_mask.sum()
                if n_invalid > 0:
                    # For critical columns, remove rows; for others, clip
                    critical_cols = ["pickup_lat", "pickup_lng", "dropoff_lat", "dropoff_lng"]
                    if col in critical_cols:
                        df = df[~invalid_mask]
                        logger.warning(f"Removed {n_invalid} rows with invalid {col}")
                    else:
                        df[col] = df[col].clip(min_val, max_val)
                        logger.debug(f"Clipped {n_invalid} values in {col}")

        rows_removed = rows_before - len(df)
        self.cleaning_stats["invalid_range_rows_removed"] = rows_removed

        return df

    def _handle_outliers(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Handle outliers in specified column.

        Uses IQR method by default for robust outlier detection.
        """
        if col not in df.columns:
            return df

        if self.outlier_method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR
        elif self.outlier_method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - self.outlier_threshold * std
            upper_bound = mean + self.outlier_threshold * std
        else:  # bounds method - use predefined ranges
            lower_bound, upper_bound = self.VALID_RANGES.get(col, (df[col].min(), df[col].max()))

        # Clip outliers instead of removing (preserve sample size)
        n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if n_outliers > 0:
            df[col] = df[col].clip(lower_bound, upper_bound)
            logger.info(f"Clipped {n_outliers} outliers in {col} to [{lower_bound:.2f}, {upper_bound:.2f}]")
            self.cleaning_stats[f"{col}_outliers_clipped"] = n_outliers

        return df

    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get summary of cleaning operations performed."""
        return self.cleaning_stats


class DataValidator:
    """
    Validates data quality and schema compliance.

    Can be used before training to ensure data meets requirements.
    """

    @staticmethod
    def validate_schema(
        df: pd.DataFrame,
        required_columns: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Check if dataframe has all required columns.

        Returns:
            (is_valid, missing_columns)
        """
        missing = [col for col in required_columns if col not in df.columns]
        return len(missing) == 0, missing

    @staticmethod
    def validate_target(
        df: pd.DataFrame,
        target_col: str = "actual_eta_minutes",
        min_val: float = 1,
        max_val: float = 180,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate target variable distribution.

        Returns:
            (is_valid, stats_dict)
        """
        if target_col not in df.columns:
            return False, {"error": f"Target column {target_col} not found"}

        target = df[target_col]
        stats = {
            "count": len(target),
            "mean": target.mean(),
            "std": target.std(),
            "min": target.min(),
            "max": target.max(),
            "null_count": target.isnull().sum(),
            "in_range_pct": ((target >= min_val) & (target <= max_val)).mean() * 100,
        }

        is_valid = (
            stats["null_count"] == 0 and
            stats["in_range_pct"] >= 95 and  # At least 95% in valid range
            stats["std"] > 0  # Some variance exists
        )

        return is_valid, stats

    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality check.

        Returns dictionary with quality metrics.
        """
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "missing_values": {},
            "column_types": {},
            "numeric_stats": {},
        }

        # Missing values per column
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                report["missing_values"][col] = {
                    "count": null_count,
                    "pct": null_count / len(df) * 100,
                }

        # Column types
        report["column_types"] = df.dtypes.astype(str).to_dict()

        # Numeric column stats
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            report["numeric_stats"][col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "skew": df[col].skew(),
            }

        return report


def clean_data(
    input_path: str,
    output_path: str,
    outlier_method: str = "iqr",
) -> pd.DataFrame:
    """
    Main function to clean raw data and save processed version.

    Args:
        input_path: Path to raw data CSV
        output_path: Path to save cleaned data
        outlier_method: Method for outlier detection

    Returns:
        Cleaned dataframe
    """
    # Load raw data
    df = pd.read_csv(input_path, parse_dates=["order_timestamp"])
    logger.info(f"Loaded {len(df)} rows from {input_path}")

    # Initialize cleaner
    cleaner = DataCleaner(outlier_method=outlier_method)

    # Clean data
    df_clean = cleaner.clean(df)

    # Validate
    validator = DataValidator()
    is_valid, stats = validator.validate_target(df_clean)
    logger.info(f"Target validation: {'PASSED' if is_valid else 'FAILED'}")
    logger.info(f"Target stats: {stats}")

    # Save cleaned data
    df_clean.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")

    # Print cleaning report
    report = cleaner.get_cleaning_report()
    logger.info(f"Cleaning report: {report}")

    return df_clean


if __name__ == "__main__":
    clean_data(
        input_path="data/raw/delivery_orders.csv",
        output_path="data/processed/delivery_orders_clean.csv",
    )
