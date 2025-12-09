"""
Load Testing with Locust

Tests API performance under load:
1. Sustained load (100 req/s target)
2. Spike testing
3. Latency distribution

Run with: locust -f tests/load_test.py --host=http://localhost:8000
"""

import random
from locust import HttpUser, task, between, constant_throughput


class DeliveryETAUser(HttpUser):
    """
    Simulates a user making delivery ETA prediction requests.

    Behavior:
    - Primarily makes real-time predictions (80%)
    - Occasionally checks health (10%)
    - Sometimes makes batch predictions (10%)
    """

    # Wait 0.5-1.5 seconds between tasks (simulates real usage)
    wait_time = between(0.5, 1.5)

    # San Francisco area coordinates
    SF_LAT_RANGE = (37.70, 37.85)
    SF_LNG_RANGE = (-122.52, -122.35)

    def _random_coordinates(self):
        """Generate random coordinates in SF area."""
        return {
            "pickup_lat": random.uniform(*self.SF_LAT_RANGE),
            "pickup_lng": random.uniform(*self.SF_LNG_RANGE),
            "dropoff_lat": random.uniform(*self.SF_LAT_RANGE),
            "dropoff_lng": random.uniform(*self.SF_LNG_RANGE),
        }

    def _random_prediction_request(self):
        """Generate a random prediction request."""
        coords = self._random_coordinates()

        return {
            **coords,
            "restaurant_id": f"R{random.randint(1, 500):04d}",
            "zone_id": random.randint(1, 8),
            "restaurant_type": random.choice(["fast_food", "casual", "fine_dining", "pizza", "asian"]),
            "traffic_multiplier": random.uniform(0.8, 1.8),
            "weather_condition": random.choice(["clear", "cloudy", "rain"]),
            "prep_time_minutes": random.uniform(5, 30),
            "rider_avg_speed_kmh": random.uniform(15, 35),
        }

    @task(80)
    def predict_realtime(self):
        """Make a real-time prediction request."""
        payload = self._random_prediction_request()

        with self.client.post(
            "/predict/realtime",
            json=payload,
            catch_response=True,
            name="/predict/realtime"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Validate response
                if "predicted_eta_minutes" not in data:
                    response.failure("Missing predicted_eta_minutes in response")
                elif data["predicted_eta_minutes"] <= 0:
                    response.failure("Invalid prediction value")
                else:
                    response.success()
            elif response.status_code == 503:
                response.failure("Service unavailable")
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(10)
    def health_check(self):
        """Check service health."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure(f"Unhealthy status: {data.get('status')}")
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(10)
    def predict_batch(self):
        """Make a batch prediction request."""
        # Create batch of 5-10 orders
        n_orders = random.randint(5, 10)
        orders = [self._random_prediction_request() for _ in range(n_orders)]

        with self.client.post(
            "/predict/batch",
            json={"orders": orders},
            catch_response=True,
            name="/predict/batch"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("total_count") != n_orders:
                    response.failure(f"Expected {n_orders} predictions, got {data.get('total_count')}")
                else:
                    response.success()
            else:
                response.failure(f"Batch prediction failed: {response.status_code}")


class HighThroughputUser(HttpUser):
    """
    User class for high-throughput testing.

    Targets constant throughput for stress testing.
    """

    # Target: 10 requests per second per user
    wait_time = constant_throughput(10)

    def _random_coordinates(self):
        return {
            "pickup_lat": random.uniform(37.70, 37.85),
            "pickup_lng": random.uniform(-122.52, -122.35),
            "dropoff_lat": random.uniform(37.70, 37.85),
            "dropoff_lng": random.uniform(-122.52, -122.35),
        }

    @task
    def predict_minimal(self):
        """Make minimal prediction request for max throughput."""
        payload = self._random_coordinates()

        self.client.post("/predict/realtime", json=payload, name="/predict/realtime")


class SpikeTestUser(HttpUser):
    """
    User class for spike testing.

    Simulates sudden bursts of traffic.
    """

    wait_time = between(0.1, 0.3)  # Very short wait for spike

    def _random_coordinates(self):
        return {
            "pickup_lat": random.uniform(37.70, 37.85),
            "pickup_lng": random.uniform(-122.52, -122.35),
            "dropoff_lat": random.uniform(37.70, 37.85),
            "dropoff_lng": random.uniform(-122.52, -122.35),
        }

    @task
    def predict(self):
        """Rapid prediction requests."""
        payload = self._random_coordinates()
        self.client.post("/predict/realtime", json=payload)


# =============================================================================
# Custom load shapes (advanced usage)
# =============================================================================

from locust import LoadTestShape


class StepLoadShape(LoadTestShape):
    """
    Step load pattern for gradual ramp-up testing.

    Increases users in steps:
    - 0-60s: 10 users
    - 60-120s: 25 users
    - 120-180s: 50 users
    - 180-240s: 100 users
    - 240-300s: ramp down
    """

    step_time = 60  # seconds per step
    step_load = 10  # users to add per step
    spawn_rate = 10  # users per second spawn rate
    time_limit = 300  # total test duration

    def tick(self):
        run_time = self.get_run_time()

        if run_time > self.time_limit:
            return None

        current_step = min(run_time // self.step_time, 4)

        if current_step < 4:
            # Ramp up
            users = (current_step + 1) * self.step_load
        else:
            # Ramp down
            remaining = self.time_limit - run_time
            users = max(0, int(remaining / self.step_time * self.step_load))

        return (users, self.spawn_rate)


class ConstantArrivalRate(LoadTestShape):
    """
    Constant arrival rate shape.

    Maintains a constant request rate regardless of response time.
    Useful for testing at specific throughput levels (e.g., 100 req/s).
    """

    target_users = 20
    spawn_rate = 5
    run_time = 300

    def tick(self):
        if self.get_run_time() > self.run_time:
            return None
        return (self.target_users, self.spawn_rate)


# =============================================================================
# Run configurations
# =============================================================================

"""
Example run commands:

# Basic load test (10 users, 1 spawn rate)
locust -f tests/load_test.py --host=http://localhost:8000 -u 10 -r 1 --run-time 60s --headless

# High throughput test (100 users)
locust -f tests/load_test.py --host=http://localhost:8000 -u 100 -r 10 --run-time 120s --headless

# Web UI mode (interactive)
locust -f tests/load_test.py --host=http://localhost:8000

# Specific user class
locust -f tests/load_test.py --host=http://localhost:8000 HighThroughputUser -u 10

# With custom shape
locust -f tests/load_test.py --host=http://localhost:8000 --headless --run-time 5m

# Generate HTML report
locust -f tests/load_test.py --host=http://localhost:8000 -u 50 -r 5 --run-time 120s --headless --html=reports/load_test_report.html
"""
