-- =============================================================================
-- Database Initialization Script
-- =============================================================================
-- Creates tables for:
-- 1. delivery_orders - Historical order data
-- 2. restaurants - Restaurant metadata
-- 3. riders - Rider profiles
-- 4. predictions - Prediction logs for monitoring
-- 5. MLflow tables (created automatically by MLflow)

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ---------------------------------------------------------------------------
-- Delivery Orders Table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS delivery_orders (
    order_id VARCHAR(20) PRIMARY KEY,
    order_timestamp TIMESTAMP NOT NULL,
    restaurant_id VARCHAR(20) NOT NULL,
    zone_id INTEGER NOT NULL,
    restaurant_type VARCHAR(50),
    pickup_lat DECIMAL(10, 7) NOT NULL,
    pickup_lng DECIMAL(10, 7) NOT NULL,
    dropoff_lat DECIMAL(10, 7) NOT NULL,
    dropoff_lng DECIMAL(10, 7) NOT NULL,
    rider_id VARCHAR(20),
    rider_avg_speed_kmh DECIMAL(5, 2),
    rider_reliability DECIMAL(4, 3),
    rider_completed_deliveries INTEGER,
    haversine_distance_km DECIMAL(6, 3),
    hour_of_day INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    is_lunch_rush BOOLEAN,
    is_dinner_rush BOOLEAN,
    traffic_multiplier DECIMAL(4, 2),
    weather_condition VARCHAR(20),
    weather_multiplier DECIMAL(4, 2),
    prep_time_minutes DECIMAL(5, 2),
    actual_eta_minutes DECIMAL(6, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_orders_timestamp ON delivery_orders(order_timestamp);
CREATE INDEX IF NOT EXISTS idx_orders_restaurant ON delivery_orders(restaurant_id);
CREATE INDEX IF NOT EXISTS idx_orders_zone ON delivery_orders(zone_id);
CREATE INDEX IF NOT EXISTS idx_orders_rider ON delivery_orders(rider_id);

-- ---------------------------------------------------------------------------
-- Restaurants Table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS restaurants (
    restaurant_id VARCHAR(20) PRIMARY KEY,
    zone_id INTEGER NOT NULL,
    zone_name VARCHAR(50),
    lat DECIMAL(10, 7) NOT NULL,
    lng DECIMAL(10, 7) NOT NULL,
    type VARCHAR(50),
    base_prep_time DECIMAL(5, 2),
    prep_std DECIMAL(4, 2),
    avg_rating DECIMAL(3, 2),
    order_volume VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_restaurants_zone ON restaurants(zone_id);
CREATE INDEX IF NOT EXISTS idx_restaurants_type ON restaurants(type);

-- ---------------------------------------------------------------------------
-- Riders Table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS riders (
    rider_id VARCHAR(20) PRIMARY KEY,
    avg_speed_kmh DECIMAL(5, 2),
    reliability_score DECIMAL(4, 3),
    completed_deliveries INTEGER,
    avg_rating DECIMAL(3, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
-- Predictions Log Table (for monitoring)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prediction_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id VARCHAR(20) NOT NULL,
    request_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    pickup_lat DECIMAL(10, 7) NOT NULL,
    pickup_lng DECIMAL(10, 7) NOT NULL,
    dropoff_lat DECIMAL(10, 7) NOT NULL,
    dropoff_lng DECIMAL(10, 7) NOT NULL,
    restaurant_id VARCHAR(20),
    zone_id INTEGER,
    predicted_eta_minutes DECIMAL(6, 2) NOT NULL,
    confidence_lower DECIMAL(6, 2),
    confidence_upper DECIMAL(6, 2),
    actual_eta_minutes DECIMAL(6, 2),  -- Updated later if feedback received
    model_version VARCHAR(20),
    latency_ms DECIMAL(8, 2),
    features_json JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON prediction_logs(request_timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON prediction_logs(model_version);

-- ---------------------------------------------------------------------------
-- Zone Statistics Table (for caching)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS zone_statistics (
    zone_id INTEGER PRIMARY KEY,
    eta_mean DECIMAL(6, 2),
    eta_median DECIMAL(6, 2),
    eta_std DECIMAL(6, 2),
    order_count INTEGER,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
-- Model Registry Table (supplement to MLflow)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    stage VARCHAR(20) DEFAULT 'staging',
    mae DECIMAL(6, 4),
    rmse DECIMAL(6, 4),
    r2 DECIMAL(6, 4),
    artifact_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deployed_at TIMESTAMP,
    UNIQUE(model_name, version)
);

-- ---------------------------------------------------------------------------
-- Drift Detection Results Table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS drift_reports (
    id SERIAL PRIMARY KEY,
    report_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    drift_detected BOOLEAN,
    drift_share DECIMAL(4, 3),
    drifted_columns TEXT[],
    target_drift_detected BOOLEAN,
    mae DECIMAL(6, 4),
    report_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
-- Functions and Triggers
-- ---------------------------------------------------------------------------

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_restaurants_updated_at
    BEFORE UPDATE ON restaurants
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_riders_updated_at
    BEFORE UPDATE ON riders
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ---------------------------------------------------------------------------
-- Sample Zone Statistics (initial values)
-- ---------------------------------------------------------------------------
INSERT INTO zone_statistics (zone_id, eta_mean, eta_median, eta_std, order_count)
VALUES
    (1, 25.5, 24.0, 8.2, 15000),
    (2, 28.3, 26.5, 9.1, 12000),
    (3, 23.1, 22.0, 7.5, 8000),
    (4, 26.8, 25.5, 8.8, 18000),
    (5, 24.2, 23.0, 7.9, 6000),
    (6, 22.5, 21.0, 6.8, 5000),
    (7, 25.0, 24.0, 8.0, 7000),
    (8, 30.5, 29.0, 10.2, 9000)
ON CONFLICT (zone_id) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
