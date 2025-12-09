"""
Data Ingestion Module

Handles loading data from multiple sources:
1. PostgreSQL database (historical orders via SQLAlchemy)
2. CSV files (batch data)
3. Google Maps API (real-time traffic/routing data)
4. REST APIs (external data sources)

Design Philosophy:
- Abstract data sources behind clean interfaces
- Support both batch (training) and real-time (inference) ingestion
- Handle failures gracefully with retries and fallbacks
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from tenacity import retry, stop_after_attempt, wait_exponential

# Optional Google Maps import
try:
    import googlemaps
    GOOGLE_MAPS_AVAILABLE = True
except ImportError:
    GOOGLE_MAPS_AVAILABLE = False
    logger.warning("googlemaps package not installed. Google Maps features disabled.")

logger.add("logs/ingestion.log", rotation="10 MB")


class DatabaseIngester:
    """
    Handles data ingestion from PostgreSQL database.

    Uses SQLAlchemy for ORM-agnostic database access.
    Supports connection pooling for production workloads.
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 5,
        max_overflow: int = 10,
    ):
        """
        Initialize database connection.

        Args:
            connection_string: PostgreSQL connection string
            pool_size: Connection pool size
            max_overflow: Max connections above pool_size
        """
        self.connection_string = connection_string
        self.engine: Optional[Engine] = None
        self.pool_size = pool_size
        self.max_overflow = max_overflow

    def connect(self) -> Engine:
        """Create database engine with connection pooling."""
        if self.engine is None:
            self.engine = create_engine(
                self.connection_string,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True,  # Verify connections before use
            )
            logger.info("Database connection established")
        return self.engine

    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            logger.info("Database connection closed")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def load_orders(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load historical orders from database.

        Args:
            start_date: Filter orders after this date
            end_date: Filter orders before this date
            limit: Maximum number of rows to return

        Returns:
            DataFrame with order data
        """
        engine = self.connect()

        query = """
        SELECT
            order_id,
            order_timestamp,
            restaurant_id,
            zone_id,
            restaurant_type,
            pickup_lat,
            pickup_lng,
            dropoff_lat,
            dropoff_lng,
            rider_id,
            rider_avg_speed_kmh,
            rider_reliability,
            rider_completed_deliveries,
            haversine_distance_km,
            hour_of_day,
            day_of_week,
            is_weekend,
            is_lunch_rush,
            is_dinner_rush,
            traffic_multiplier,
            weather_condition,
            weather_multiplier,
            prep_time_minutes,
            actual_eta_minutes
        FROM delivery_orders
        WHERE 1=1
        """

        params: Dict[str, Any] = {}

        if start_date:
            query += " AND order_timestamp >= :start_date"
            params["start_date"] = start_date

        if end_date:
            query += " AND order_timestamp <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY order_timestamp"

        if limit:
            query += f" LIMIT {limit}"

        logger.info(f"Executing query with params: {params}")

        df = pd.read_sql(text(query), engine, params=params)
        logger.info(f"Loaded {len(df)} orders from database")

        return df

    def load_restaurants(self) -> pd.DataFrame:
        """Load restaurant metadata."""
        engine = self.connect()
        query = "SELECT * FROM restaurants"
        return pd.read_sql(query, engine)

    def load_riders(self) -> pd.DataFrame:
        """Load rider metadata."""
        engine = self.connect()
        query = "SELECT * FROM riders"
        return pd.read_sql(query, engine)


class CSVIngester:
    """
    Handles data ingestion from CSV files.

    Simple but effective for batch processing and local development.
    """

    @staticmethod
    def load_orders(
        filepath: str,
        parse_dates: bool = True,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load orders from CSV file.

        Args:
            filepath: Path to CSV file
            parse_dates: Whether to parse date columns
            nrows: Number of rows to read (None for all)

        Returns:
            DataFrame with order data
        """
        date_cols = ["order_timestamp"] if parse_dates else None

        df = pd.read_csv(
            filepath,
            parse_dates=date_cols,
            nrows=nrows,
        )

        logger.info(f"Loaded {len(df)} orders from {filepath}")
        return df

    @staticmethod
    def load_restaurants(filepath: str) -> pd.DataFrame:
        """Load restaurant data from CSV."""
        return pd.read_csv(filepath)

    @staticmethod
    def load_riders(filepath: str) -> pd.DataFrame:
        """Load rider data from CSV."""
        return pd.read_csv(filepath)


class GoogleMapsClient:
    """
    Google Maps API client for real-time traffic and routing data.

    Provides:
    - Distance Matrix API: Travel time/distance between points
    - Directions API: Detailed route with traffic-aware ETA

    Why Google Maps?
    - Industry standard for routing with real-time traffic
    - Provides baseline ETA that our ML model can improve upon
    - Handles complex routing (one-way streets, turn restrictions, etc.)
    """

    def __init__(self, api_key: str, timeout: int = 10):
        """
        Initialize Google Maps client.

        Args:
            api_key: Google Maps API key
            timeout: Request timeout in seconds
        """
        if not GOOGLE_MAPS_AVAILABLE:
            raise ImportError("googlemaps package required. Install with: pip install googlemaps")

        self.client = googlemaps.Client(key=api_key, timeout=timeout)
        self.api_key = api_key
        logger.info("Google Maps client initialized")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    def get_directions_eta(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
        departure_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get ETA from Google Directions API with traffic.

        Args:
            origin_lat, origin_lng: Pickup coordinates
            dest_lat, dest_lng: Dropoff coordinates
            departure_time: When to calculate traffic for (default: now)

        Returns:
            Dict with duration_minutes, distance_km, traffic_duration_minutes
        """
        if departure_time is None:
            departure_time = datetime.now()

        origin = (origin_lat, origin_lng)
        destination = (dest_lat, dest_lng)

        try:
            result = self.client.directions(
                origin=origin,
                destination=destination,
                mode="driving",
                departure_time=departure_time,
                traffic_model="best_guess",
            )

            if not result:
                logger.warning("No route found between points")
                return self._fallback_estimate(origin_lat, origin_lng, dest_lat, dest_lng)

            leg = result[0]["legs"][0]

            return {
                "duration_minutes": leg["duration"]["value"] / 60,
                "duration_in_traffic_minutes": leg.get("duration_in_traffic", leg["duration"])["value"] / 60,
                "distance_km": leg["distance"]["value"] / 1000,
                "source": "google_maps",
            }

        except Exception as e:
            logger.error(f"Google Maps API error: {e}")
            return self._fallback_estimate(origin_lat, origin_lng, dest_lat, dest_lng)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    def get_distance_matrix(
        self,
        origins: List[tuple],
        destinations: List[tuple],
        departure_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get distance matrix for multiple origin-destination pairs.

        Useful for batch processing or finding optimal assignments.

        Args:
            origins: List of (lat, lng) tuples
            destinations: List of (lat, lng) tuples
            departure_time: When to calculate traffic for

        Returns:
            DataFrame with distance and duration for each pair
        """
        if departure_time is None:
            departure_time = datetime.now()

        result = self.client.distance_matrix(
            origins=origins,
            destinations=destinations,
            mode="driving",
            departure_time=departure_time,
            traffic_model="best_guess",
        )

        rows = []
        for i, origin in enumerate(result["origin_addresses"]):
            for j, dest in enumerate(result["destination_addresses"]):
                element = result["rows"][i]["elements"][j]
                if element["status"] == "OK":
                    rows.append({
                        "origin_idx": i,
                        "dest_idx": j,
                        "origin_address": origin,
                        "dest_address": dest,
                        "distance_km": element["distance"]["value"] / 1000,
                        "duration_minutes": element["duration"]["value"] / 60,
                        "duration_in_traffic_minutes": element.get("duration_in_traffic", element["duration"])["value"] / 60,
                    })

        return pd.DataFrame(rows)

    def _fallback_estimate(
        self,
        lat1: float,
        lng1: float,
        lat2: float,
        lng2: float,
    ) -> Dict[str, Any]:
        """
        Fallback estimate when Google Maps is unavailable.

        Uses haversine distance with assumed average speed.
        """
        from geopy.distance import geodesic

        distance_km = geodesic((lat1, lng1), (lat2, lng2)).kilometers
        # Assume average speed of 25 km/h in urban areas
        duration_minutes = (distance_km / 25) * 60

        return {
            "duration_minutes": duration_minutes,
            "duration_in_traffic_minutes": duration_minutes * 1.2,  # 20% buffer
            "distance_km": distance_km,
            "source": "haversine_fallback",
        }


class ExternalAPIClient:
    """
    Generic client for external REST APIs.

    Can be extended for:
    - Weather APIs (OpenWeatherMap, etc.)
    - Traffic APIs (TomTom, HERE, etc.)
    - Special events APIs (concerts, sports, etc.)
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 10):
        """
        Initialize API client.

        Args:
            base_url: Base URL for the API
            api_key: Optional API key
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make GET request to API."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self.client.get(url, params=params, headers=headers)
        response.raise_for_status()

        return response.json()

    def close(self):
        """Close HTTP client."""
        self.client.close()


class DataLoader:
    """
    Unified data loader that abstracts different data sources.

    Provides a single interface for loading data regardless of source.
    Automatically handles source selection based on configuration.
    """

    def __init__(
        self,
        db_connection: Optional[str] = None,
        csv_path: Optional[str] = None,
        google_maps_key: Optional[str] = None,
    ):
        """
        Initialize data loader with available sources.

        Args:
            db_connection: Database connection string
            csv_path: Path to CSV data
            google_maps_key: Google Maps API key
        """
        self.db_ingester = DatabaseIngester(db_connection) if db_connection else None
        self.csv_path = csv_path
        self.google_client = GoogleMapsClient(google_maps_key) if google_maps_key else None

    def load_training_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load training data from best available source.

        Priority: Database > CSV
        """
        # Try database first
        if self.db_ingester:
            try:
                return self.db_ingester.load_orders(start_date, end_date)
            except Exception as e:
                logger.warning(f"Database load failed: {e}. Falling back to CSV.")

        # Fall back to CSV
        if self.csv_path:
            df = CSVIngester.load_orders(self.csv_path)
            if start_date:
                df = df[df["order_timestamp"] >= start_date]
            if end_date:
                df = df[df["order_timestamp"] <= end_date]
            return df

        raise ValueError("No data source available")

    def get_realtime_eta_baseline(
        self,
        pickup_lat: float,
        pickup_lng: float,
        dropoff_lat: float,
        dropoff_lng: float,
    ) -> Dict[str, Any]:
        """
        Get real-time ETA baseline from Google Maps.

        This provides the traffic-aware routing ETA that our ML model
        can then adjust based on historical patterns.
        """
        if self.google_client:
            return self.google_client.get_directions_eta(
                pickup_lat, pickup_lng, dropoff_lat, dropoff_lng
            )
        else:
            # Fallback to simple haversine estimate
            from geopy.distance import geodesic
            distance = geodesic(
                (pickup_lat, pickup_lng),
                (dropoff_lat, dropoff_lng)
            ).kilometers
            duration = (distance / 25) * 60  # Assume 25 km/h
            return {
                "duration_minutes": duration,
                "duration_in_traffic_minutes": duration * 1.2,
                "distance_km": distance,
                "source": "haversine_fallback",
            }

    def close(self):
        """Clean up resources."""
        if self.db_ingester:
            self.db_ingester.close()
