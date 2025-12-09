"""
Synthetic Data Generation Module

Generates realistic delivery order data for training and testing.
Simulates real-world patterns including:
- Geographic distribution across urban zones
- Time-based patterns (rush hours, weekends)
- Restaurant-specific preparation times
- Traffic conditions affecting delivery times
- Weather impact on ETA

This synthetic data mimics real delivery platforms like Uber Eats/DoorDash.
"""

import random
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from loguru import logger

# Configure logger
logger.add("logs/data_generation.log", rotation="10 MB")


class DeliveryDataGenerator:
    """
    Generates synthetic delivery order data with realistic patterns.

    The generator creates data that captures real-world delivery dynamics:
    - Urban zone clustering (restaurants concentrated in business districts)
    - Time-of-day effects (lunch/dinner rush)
    - Day-of-week patterns (weekends differ from weekdays)
    - Restaurant preparation variability
    - Traffic multipliers based on time and conditions
    - Weather impact simulation
    """

    # San Francisco Bay Area bounding box (can be configured for other cities)
    DEFAULT_BOUNDS = {
        "lat_min": 37.70,
        "lat_max": 37.85,
        "lng_min": -122.52,
        "lng_max": -122.35,
    }

    # Restaurant zones (business districts, neighborhoods)
    ZONES = [
        {"id": 1, "name": "Downtown", "lat": 37.7879, "lng": -122.4074, "radius": 0.02},
        {"id": 2, "name": "Mission", "lat": 37.7599, "lng": -122.4148, "radius": 0.015},
        {"id": 3, "name": "Marina", "lat": 37.8025, "lng": -122.4382, "radius": 0.012},
        {"id": 4, "name": "SOMA", "lat": 37.7785, "lng": -122.3950, "radius": 0.018},
        {"id": 5, "name": "Castro", "lat": 37.7609, "lng": -122.4350, "radius": 0.01},
        {"id": 6, "name": "North Beach", "lat": 37.8060, "lng": -122.4103, "radius": 0.008},
        {"id": 7, "name": "Hayes Valley", "lat": 37.7759, "lng": -122.4245, "radius": 0.008},
        {"id": 8, "name": "Sunset", "lat": 37.7533, "lng": -122.4869, "radius": 0.025},
    ]

    # Restaurant types with prep time characteristics
    RESTAURANT_TYPES = [
        {"type": "fast_food", "base_prep": 8, "std": 2},
        {"type": "casual", "base_prep": 15, "std": 4},
        {"type": "fine_dining", "base_prep": 25, "std": 6},
        {"type": "pizza", "base_prep": 18, "std": 3},
        {"type": "asian", "base_prep": 12, "std": 3},
        {"type": "mexican", "base_prep": 10, "std": 2},
        {"type": "coffee_shop", "base_prep": 5, "std": 1},
    ]

    # Weather conditions affecting delivery
    WEATHER_CONDITIONS = [
        {"condition": "clear", "multiplier": 1.0, "probability": 0.6},
        {"condition": "cloudy", "multiplier": 1.05, "probability": 0.2},
        {"condition": "rain", "multiplier": 1.3, "probability": 0.12},
        {"condition": "heavy_rain", "multiplier": 1.5, "probability": 0.05},
        {"condition": "fog", "multiplier": 1.15, "probability": 0.03},
    ]

    def __init__(
        self,
        n_restaurants: int = 500,
        n_riders: int = 200,
        random_seed: int = 42,
        bounds: Optional[dict] = None,
    ):
        """
        Initialize the data generator.

        Args:
            n_restaurants: Number of restaurants to simulate
            n_riders: Number of delivery riders
            random_seed: Random seed for reproducibility
            bounds: Geographic bounds dict with lat_min, lat_max, lng_min, lng_max
        """
        self.n_restaurants = n_restaurants
        self.n_riders = n_riders
        self.random_seed = random_seed
        self.bounds = bounds or self.DEFAULT_BOUNDS

        np.random.seed(random_seed)
        random.seed(random_seed)

        # Generate restaurant and rider profiles
        self.restaurants = self._generate_restaurants()
        self.riders = self._generate_riders()

        logger.info(
            f"Initialized generator: {n_restaurants} restaurants, {n_riders} riders"
        )

    def _generate_restaurants(self) -> pd.DataFrame:
        """Generate restaurant profiles clustered around zones."""
        restaurants = []

        for i in range(self.n_restaurants):
            # Select a zone (weighted by zone size)
            zone = random.choice(self.ZONES)
            rest_type = random.choice(self.RESTAURANT_TYPES)

            # Position restaurant near zone center with noise
            lat = np.random.normal(zone["lat"], zone["radius"])
            lng = np.random.normal(zone["lng"], zone["radius"])

            # Clip to bounds
            lat = np.clip(lat, self.bounds["lat_min"], self.bounds["lat_max"])
            lng = np.clip(lng, self.bounds["lng_min"], self.bounds["lng_max"])

            # Restaurant-specific prep time variation
            prep_offset = np.random.normal(0, 2)

            restaurants.append({
                "restaurant_id": f"R{i+1:04d}",
                "zone_id": zone["id"],
                "zone_name": zone["name"],
                "lat": lat,
                "lng": lng,
                "type": rest_type["type"],
                "base_prep_time": rest_type["base_prep"] + prep_offset,
                "prep_std": rest_type["std"],
                "avg_rating": np.random.uniform(3.5, 5.0),
                "order_volume": np.random.choice(["low", "medium", "high"]),
            })

        return pd.DataFrame(restaurants)

    def _generate_riders(self) -> pd.DataFrame:
        """Generate rider profiles with performance characteristics."""
        riders = []

        for i in range(self.n_riders):
            # Rider performance varies (experience, vehicle type, etc.)
            base_speed = np.random.uniform(15, 35)  # km/h average
            reliability = np.random.uniform(0.85, 0.99)

            riders.append({
                "rider_id": f"D{i+1:04d}",
                "avg_speed_kmh": base_speed,
                "reliability_score": reliability,
                "completed_deliveries": int(np.random.exponential(200)),
                "avg_rating": np.random.uniform(4.0, 5.0),
            })

        return pd.DataFrame(riders)

    def _get_traffic_multiplier(self, hour: int, day_of_week: int) -> float:
        """
        Calculate traffic multiplier based on time.

        Models real traffic patterns:
        - Morning rush (7-9 AM): 1.4-1.6x
        - Lunch rush (12-1 PM): 1.3x
        - Evening rush (5-7 PM): 1.5-1.8x
        - Late night (10 PM - 6 AM): 0.8-0.9x
        - Weekends: Generally lower
        """
        # Base traffic by hour
        hourly_traffic = {
            0: 0.8, 1: 0.75, 2: 0.7, 3: 0.7, 4: 0.75, 5: 0.85,
            6: 1.0, 7: 1.4, 8: 1.6, 9: 1.3, 10: 1.1, 11: 1.2,
            12: 1.35, 13: 1.25, 14: 1.1, 15: 1.1, 16: 1.3, 17: 1.6,
            18: 1.7, 19: 1.5, 20: 1.2, 21: 1.0, 22: 0.9, 23: 0.85,
        }

        base = hourly_traffic.get(hour, 1.0)

        # Weekend adjustment (less traffic)
        if day_of_week >= 5:  # Saturday, Sunday
            base *= 0.85

        # Add some randomness
        noise = np.random.uniform(0.9, 1.1)

        return base * noise

    def _get_weather(self) -> Tuple[str, float]:
        """Sample weather condition and return (condition, multiplier)."""
        probs = [w["probability"] for w in self.WEATHER_CONDITIONS]
        weather = np.random.choice(self.WEATHER_CONDITIONS, p=probs)
        return weather["condition"], weather["multiplier"]

    def _calculate_haversine(
        self, lat1: float, lng1: float, lat2: float, lng2: float
    ) -> float:
        """Calculate haversine distance in kilometers."""
        return geodesic((lat1, lng1), (lat2, lng2)).kilometers

    def _calculate_actual_eta(
        self,
        distance_km: float,
        prep_time: float,
        traffic_mult: float,
        weather_mult: float,
        rider_speed: float,
        rider_reliability: float,
    ) -> float:
        """
        Calculate realistic actual ETA with all factors.

        Components:
        1. Restaurant prep time
        2. Travel time = distance / rider_speed * traffic * weather
        3. Fixed overhead (parking, finding address, etc.): ~3 min
        4. Rider-specific reliability factor
        """
        # Travel time in minutes
        travel_time = (distance_km / rider_speed) * 60  # Convert to minutes

        # Apply multipliers
        travel_time *= traffic_mult * weather_mult

        # Fixed overhead (finding parking, walking to door, etc.)
        overhead = np.random.uniform(2, 5)

        # Rider reliability affects consistency
        reliability_factor = 1 + (1 - rider_reliability) * np.random.uniform(-0.5, 0.5)

        # Total ETA
        total_eta = (prep_time + travel_time + overhead) * reliability_factor

        # Add realistic noise (real-world unpredictability)
        noise = np.random.normal(0, 2)
        total_eta += noise

        return max(5, total_eta)  # Minimum 5 minutes

    def generate_orders(
        self,
        n_orders: int = 100000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic delivery orders.

        Args:
            n_orders: Number of orders to generate
            start_date: Start of date range
            end_date: End of date range

        Returns:
            DataFrame with order data including features and target (actual_eta_minutes)
        """
        if start_date is None:
            start_date = datetime(2023, 1, 1)
        if end_date is None:
            end_date = datetime(2024, 1, 1)

        logger.info(f"Generating {n_orders} orders from {start_date} to {end_date}")

        orders = []
        date_range = (end_date - start_date).days

        for i in range(n_orders):
            if i % 10000 == 0:
                logger.info(f"Generated {i}/{n_orders} orders")

            # Random order timestamp
            random_days = random.randint(0, date_range)
            random_seconds = random.randint(0, 86400)
            order_time = start_date + timedelta(days=random_days, seconds=random_seconds)

            # Bias towards meal times (11 AM - 2 PM, 5 PM - 9 PM)
            hour = order_time.hour
            if random.random() < 0.7:  # 70% during meal hours
                if random.random() < 0.4:  # Lunch
                    hour = random.randint(11, 14)
                else:  # Dinner
                    hour = random.randint(17, 21)
                order_time = order_time.replace(hour=hour)

            # Select restaurant
            restaurant = self.restaurants.sample(1).iloc[0]

            # Generate dropoff location (customer)
            # Most deliveries are within 3-5 km
            distance_target = np.random.exponential(2.5) + 0.5
            angle = np.random.uniform(0, 2 * np.pi)

            # Convert distance to lat/lng offset (approximate)
            lat_offset = (distance_target / 111) * np.cos(angle)
            lng_offset = (distance_target / (111 * np.cos(np.radians(restaurant["lat"])))) * np.sin(angle)

            dropoff_lat = restaurant["lat"] + lat_offset
            dropoff_lng = restaurant["lng"] + lng_offset

            # Clip to bounds
            dropoff_lat = np.clip(dropoff_lat, self.bounds["lat_min"], self.bounds["lat_max"])
            dropoff_lng = np.clip(dropoff_lng, self.bounds["lng_min"], self.bounds["lng_max"])

            # Calculate actual distance
            distance_km = self._calculate_haversine(
                restaurant["lat"], restaurant["lng"], dropoff_lat, dropoff_lng
            )

            # Select rider
            rider = self.riders.sample(1).iloc[0]

            # Get time factors
            traffic_mult = self._get_traffic_multiplier(order_time.hour, order_time.weekday())
            weather, weather_mult = self._get_weather()

            # Calculate prep time for this order
            prep_time = np.random.normal(restaurant["base_prep_time"], restaurant["prep_std"])
            prep_time = max(3, prep_time)  # Minimum 3 minutes

            # Calculate actual ETA
            actual_eta = self._calculate_actual_eta(
                distance_km=distance_km,
                prep_time=prep_time,
                traffic_mult=traffic_mult,
                weather_mult=weather_mult,
                rider_speed=rider["avg_speed_kmh"],
                rider_reliability=rider["reliability_score"],
            )

            # Order details
            order = {
                "order_id": f"O{i+1:07d}",
                "order_timestamp": order_time,
                "restaurant_id": restaurant["restaurant_id"],
                "zone_id": restaurant["zone_id"],
                "restaurant_type": restaurant["type"],
                "pickup_lat": restaurant["lat"],
                "pickup_lng": restaurant["lng"],
                "dropoff_lat": dropoff_lat,
                "dropoff_lng": dropoff_lng,
                "rider_id": rider["rider_id"],
                "rider_avg_speed_kmh": rider["avg_speed_kmh"],
                "rider_reliability": rider["reliability_score"],
                "rider_completed_deliveries": rider["completed_deliveries"],
                "haversine_distance_km": distance_km,
                "hour_of_day": order_time.hour,
                "day_of_week": order_time.weekday(),
                "is_weekend": order_time.weekday() >= 5,
                "is_lunch_rush": 11 <= order_time.hour <= 14,
                "is_dinner_rush": 17 <= order_time.hour <= 21,
                "traffic_multiplier": traffic_mult,
                "weather_condition": weather,
                "weather_multiplier": weather_mult,
                "prep_time_minutes": prep_time,
                "actual_eta_minutes": actual_eta,  # TARGET VARIABLE
            }

            orders.append(order)

        df = pd.DataFrame(orders)

        # Add derived features
        df["month"] = df["order_timestamp"].dt.month
        df["day_of_month"] = df["order_timestamp"].dt.day
        df["week_of_year"] = df["order_timestamp"].dt.isocalendar().week

        logger.info(f"Generated {len(df)} orders. ETA stats: mean={df['actual_eta_minutes'].mean():.1f}, "
                   f"std={df['actual_eta_minutes'].std():.1f}, min={df['actual_eta_minutes'].min():.1f}, "
                   f"max={df['actual_eta_minutes'].max():.1f}")

        return df


def main():
    """Generate and save synthetic delivery data."""
    import sys
    sys.path.insert(0, ".")

    from config.settings import get_settings

    settings = get_settings()

    generator = DeliveryDataGenerator(
        n_restaurants=500,
        n_riders=200,
        random_seed=settings.random_seed,
    )

    df = generator.generate_orders(n_orders=settings.synthetic_data_rows)

    # Save to CSV
    output_path = "data/raw/delivery_orders.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} orders to {output_path}")

    # Also save restaurant and rider profiles for reference
    generator.restaurants.to_csv("data/raw/restaurants.csv", index=False)
    generator.riders.to_csv("data/raw/riders.csv", index=False)

    # Print summary stats
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Orders generated: {len(df):,}")
    print(f"\nTarget variable (actual_eta_minutes) statistics:")
    print(df["actual_eta_minutes"].describe())
    print(f"\nDistance statistics (km):")
    print(df["haversine_distance_km"].describe())
    print(f"\nData saved to: {output_path}")


if __name__ == "__main__":
    main()
