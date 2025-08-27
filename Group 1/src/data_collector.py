"""
Data Collection Module for 3D Navigation System
Collects anonymous GPS, Wi-Fi/Bluetooth, barometer, and accelerometer data from user phones.
"""

import json
import sqlite3
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class SensorData:
    """Data structure for sensor readings from mobile devices."""
    timestamp: datetime
    user_id: str  # Anonymous hashed ID
    gps_lat: float
    gps_lon: float
    gps_accuracy: float
    wifi_signals: List[Dict[str, float]]  # {ssid_hash: signal_strength}
    bluetooth_beacons: List[Dict[str, float]]  # {device_hash: signal_strength}
    barometer_pressure: float  # hPa
    accelerometer_x: float  # m/s²
    accelerometer_y: float  # m/s²
    accelerometer_z: float  # m/s²
    gyroscope_x: float  # rad/s
    gyroscope_y: float  # rad/s
    gyroscope_z: float  # rad/s
    magnetometer_x: float  # µT
    magnetometer_y: float  # µT
    magnetometer_z: float  # µT


class DataCollector:
    """Manages collection and storage of sensor data from mobile devices."""
    
    def __init__(self, db_path: str = "data/sensor_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for sensor data storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                gps_lat REAL NOT NULL,
                gps_lon REAL NOT NULL,
                gps_accuracy REAL NOT NULL,
                wifi_signals TEXT,
                bluetooth_beacons TEXT,
                barometer_pressure REAL NOT NULL,
                accelerometer_x REAL NOT NULL,
                accelerometer_y REAL NOT NULL,
                accelerometer_z REAL NOT NULL,
                gyroscope_x REAL NOT NULL,
                gyroscope_y REAL NOT NULL,
                gyroscope_z REAL NOT NULL,
                magnetometer_x REAL NOT NULL,
                magnetometer_y REAL NOT NULL,
                magnetometer_z REAL NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON sensor_readings(timestamp);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_location ON sensor_readings(gps_lat, gps_lon);
        """)
        
        conn.commit()
        conn.close()
    
    async def collect_sensor_data(self, sensor_data: SensorData) -> bool:
        """
        Store sensor data in database.
        
        Args:
            sensor_data: SensorData object containing all sensor readings
            
        Returns:
            bool: True if data stored successfully
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO sensor_readings (
                    timestamp, user_id, gps_lat, gps_lon, gps_accuracy,
                    wifi_signals, bluetooth_beacons, barometer_pressure,
                    accelerometer_x, accelerometer_y, accelerometer_z,
                    gyroscope_x, gyroscope_y, gyroscope_z,
                    magnetometer_x, magnetometer_y, magnetometer_z
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sensor_data.timestamp.isoformat(),
                sensor_data.user_id,
                sensor_data.gps_lat,
                sensor_data.gps_lon,
                sensor_data.gps_accuracy,
                json.dumps(sensor_data.wifi_signals),
                json.dumps(sensor_data.bluetooth_beacons),
                sensor_data.barometer_pressure,
                sensor_data.accelerometer_x,
                sensor_data.accelerometer_y,
                sensor_data.accelerometer_z,
                sensor_data.gyroscope_x,
                sensor_data.gyroscope_y,
                sensor_data.gyroscope_z,
                sensor_data.magnetometer_x,
                sensor_data.magnetometer_y,
                sensor_data.magnetometer_z
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error storing sensor data: {e}")
            return False
    
    def get_training_data(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieve sensor data for model training.
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            List of sensor data dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM sensor_readings ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"
            
        cursor.execute(query)
        rows = cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        data = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return data
    
    def get_location_clusters(self, radius_meters: float = 50) -> List[Tuple[float, float, int]]:
        """
        Get location clusters for graph node generation.
        
        Args:
            radius_meters: Clustering radius in meters
            
        Returns:
            List of (lat, lon, count) tuples representing location clusters
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple grid-based clustering
        grid_size = radius_meters / 111000  # Approximate degrees per meter
        
        cursor.execute("""
            SELECT 
                ROUND(gps_lat / ?, 0) * ? as lat_cluster,
                ROUND(gps_lon / ?, 0) * ? as lon_cluster,
                COUNT(*) as point_count
            FROM sensor_readings 
            GROUP BY lat_cluster, lon_cluster
            HAVING point_count > 5
            ORDER BY point_count DESC
        """, (grid_size, grid_size, grid_size, grid_size))
        
        clusters = cursor.fetchall()
        conn.close()
        
        return clusters
    
    def simulate_sensor_reading(self, lat: float, lon: float, elevation: int = 0) -> SensorData:
        """
        Simulate sensor reading for testing purposes.
        
        Args:
            lat: GPS latitude
            lon: GPS longitude
            elevation: Known elevation level for testing
            
        Returns:
            SensorData object with simulated readings
        """
        base_pressure = 1013.25  # Sea level pressure in hPa
        pressure_drop_per_meter = 0.12  # Approximate hPa drop per meter altitude
        
        # Simulate elevation-dependent pressure
        simulated_pressure = base_pressure - (elevation * 3 * pressure_drop_per_meter)  # 3m per floor
        
        return SensorData(
            timestamp=datetime.now(),
            user_id=f"sim_user_{hash(str(lat) + str(lon)) % 10000}",
            gps_lat=lat + np.random.normal(0, 0.0001),  # Add GPS noise
            gps_lon=lon + np.random.normal(0, 0.0001),
            gps_accuracy=np.random.uniform(3, 15),
            wifi_signals=[
                {"ap_hash_" + str(i): -40 - np.random.exponential(20)} 
                for i in range(np.random.randint(1, 6))
            ],
            bluetooth_beacons=[
                {"beacon_hash_" + str(i): -60 - np.random.exponential(15)} 
                for i in range(np.random.randint(0, 4))
            ],
            barometer_pressure=simulated_pressure + np.random.normal(0, 0.5),
            accelerometer_x=np.random.normal(0, 0.5),
            accelerometer_y=np.random.normal(0, 0.5),
            accelerometer_z=np.random.normal(-9.81, 0.5),  # Gravity
            gyroscope_x=np.random.normal(0, 0.1),
            gyroscope_y=np.random.normal(0, 0.1),
            gyroscope_z=np.random.normal(0, 0.1),
            magnetometer_x=np.random.normal(20, 5),
            magnetometer_y=np.random.normal(0, 5),
            magnetometer_z=np.random.normal(-45, 5)
        )


class DataPreprocessor:
    """Preprocesses raw sensor data for XGBoost model training."""
    
    @staticmethod
    def extract_features(sensor_data: Dict) -> Dict:
        """
        Extract features from raw sensor data for ML model.
        
        Args:
            sensor_data: Raw sensor data dictionary
            
        Returns:
            Dictionary of extracted features
        """
        wifi_signals = json.loads(sensor_data.get('wifi_signals', '[]'))
        bluetooth_beacons = json.loads(sensor_data.get('bluetooth_beacons', '[]'))
        
        # WiFi features
        wifi_count = len(wifi_signals)
        wifi_strengths = [signal for signals in wifi_signals for signal in signals.values()]
        wifi_max_strength = max(wifi_strengths) if wifi_strengths else -100
        wifi_avg_strength = np.mean(wifi_strengths) if wifi_strengths else -100
        wifi_std_strength = np.std(wifi_strengths) if len(wifi_strengths) > 1 else 0
        
        # Bluetooth features
        bt_count = len(bluetooth_beacons)
        bt_strengths = [signal for beacons in bluetooth_beacons for signal in beacons.values()]
        bt_max_strength = max(bt_strengths) if bt_strengths else -100
        bt_avg_strength = np.mean(bt_strengths) if bt_strengths else -100
        
        # Motion features
        accel_magnitude = np.sqrt(
            sensor_data['accelerometer_x']**2 + 
            sensor_data['accelerometer_y']**2 + 
            sensor_data['accelerometer_z']**2
        )
        
        gyro_magnitude = np.sqrt(
            sensor_data['gyroscope_x']**2 + 
            sensor_data['gyroscope_y']**2 + 
            sensor_data['gyroscope_z']**2
        )
        
        return {
            'gps_lat': sensor_data['gps_lat'],
            'gps_lon': sensor_data['gps_lon'],
            'gps_accuracy': sensor_data['gps_accuracy'],
            'barometer_pressure': sensor_data['barometer_pressure'],
            'wifi_count': wifi_count,
            'wifi_max_strength': wifi_max_strength,
            'wifi_avg_strength': wifi_avg_strength,
            'wifi_std_strength': wifi_std_strength,
            'bt_count': bt_count,
            'bt_max_strength': bt_max_strength,
            'bt_avg_strength': bt_avg_strength,
            'accel_magnitude': accel_magnitude,
            'gyro_magnitude': gyro_magnitude,
            'magnetometer_x': sensor_data['magnetometer_x'],
            'magnetometer_y': sensor_data['magnetometer_y'],
            'magnetometer_z': sensor_data['magnetometer_z']
        }


if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    # Simulate some data collection for Chongqing area
    chongqing_coords = [
        (29.5630, 106.5516, 0),  # Ground level
        (29.5631, 106.5517, 1),  # First elevated level
        (29.5632, 106.5518, 2),  # Second elevated level
    ]
    
    async def simulate_data_collection():
        for lat, lon, elevation in chongqing_coords:
            for _ in range(10):  # 10 readings per location
                sensor_reading = collector.simulate_sensor_reading(lat, lon, elevation)
                await collector.collect_sensor_data(sensor_reading)
        
        print(f"Collected simulated data for {len(chongqing_coords)} locations")
        
        # Show some statistics
        clusters = collector.get_location_clusters()
        print(f"Found {len(clusters)} location clusters")
    
    asyncio.run(simulate_data_collection())