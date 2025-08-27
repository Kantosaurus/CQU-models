"""
Configuration settings for the 3D Navigation System
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Database configuration
DATABASE_CONFIG = {
    "sensor_data_db": str(DATA_DIR / "sensor_data.db"),
    "navigation_graph": str(DATA_DIR / "navigation_graph.json"),
    "elevation_model": str(MODELS_DIR / "elevation_predictor.joblib")
}

# XGBoost model parameters
XGBOOST_CONFIG = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss"
}

# Graph building parameters
GRAPH_CONFIG = {
    "cluster_radius_meters": 50,
    "min_cluster_size": 5,
    "max_walking_distance": 200,
    "max_elevation_levels": 10
}

# Route planning defaults
ROUTE_CONFIG = {
    "max_walking_distance": 1000,
    "max_total_time": 1800,  # 30 minutes
    "max_elevation_change": 10,
    "walking_speed_ms": 1.4,  # 1.4 m/s = 5 km/h
    "default_alternatives": 3
}

# Chongqing-specific settings
CHONGQING_CONFIG = {
    "center_lat": 29.5630,
    "center_lon": 106.5516,
    "bounds": {
        "min_lat": 29.4,
        "max_lat": 29.8,
        "min_lon": 106.3,
        "max_lon": 106.8
    },
    "sea_level_pressure": 1013.25,  # hPa
    "elevation_pressure_drop": 0.12  # hPa per meter
}

# Web application settings
WEB_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "debug": True,
    "title": "3D Navigation System",
    "description": "AI-powered 3D navigation for Chongqing's multi-level terrain"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(LOGS_DIR / "navigation.log")
}

# Feature engineering settings
FEATURE_CONFIG = {
    "wifi_max_signals": 10,
    "bluetooth_max_beacons": 5,
    "pressure_smoothing_window": 5,
    "motion_magnitude_threshold": 15.0
}

# Privacy and security settings
PRIVACY_CONFIG = {
    "anonymize_user_ids": True,
    "data_retention_days": 90,
    "min_location_accuracy": 50,  # meters
    "location_blur_radius": 10   # meters
}