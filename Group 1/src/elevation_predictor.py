"""
XGBoost Elevation Prediction Model
Predicts elevation level based on multi-sensor data from mobile devices.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
from typing import Dict, List, Tuple, Optional
import sqlite3
from datetime import datetime

from data_collector import DataCollector, DataPreprocessor


class ElevationPredictor:
    """XGBoost model for predicting building elevation levels from sensor data."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        if model_path:
            self.load_model(model_path)
    
    def prepare_training_data(self, data_collector: DataCollector) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from collected sensor readings.
        
        Args:
            data_collector: DataCollector instance with sensor data
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        raw_data = data_collector.get_training_data()
        
        if not raw_data:
            raise ValueError("No training data available")
        
        features_list = []
        labels_list = []
        
        for record in raw_data:
            # Extract features using preprocessor
            features = DataPreprocessor.extract_features(record)
            
            # For demonstration, we'll derive elevation from barometer pressure
            # In real implementation, this would come from ground truth data
            base_pressure = 1013.25
            pressure_diff = base_pressure - record['barometer_pressure']
            estimated_elevation = max(0, int(pressure_diff / 0.36))  # Rough elevation estimate
            
            features_list.append(features)
            labels_list.append(min(estimated_elevation, 10))  # Cap at 10 levels
        
        features_df = pd.DataFrame(features_list)
        labels_series = pd.Series(labels_list)
        
        # Store feature names for later use
        self.feature_names = features_df.columns.tolist()
        
        return features_df, labels_series
    
    def create_synthetic_training_data(self, n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create synthetic training data for demonstration purposes.
        
        Args:
            n_samples: Number of synthetic samples to generate
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        np.random.seed(42)
        
        features_list = []
        labels_list = []
        
        # Chongqing approximate coordinates
        base_lat, base_lon = 29.5630, 106.5516
        
        for _ in range(n_samples):
            # Random elevation level (0-5 floors)
            elevation = np.random.randint(0, 6)
            
            # Generate realistic sensor data based on elevation
            lat = base_lat + np.random.normal(0, 0.01)
            lon = base_lon + np.random.normal(0, 0.01)
            
            # Barometer: higher elevation = lower pressure
            base_pressure = 1013.25
            pressure = base_pressure - (elevation * 3 * 0.12) + np.random.normal(0, 1)
            
            # WiFi: higher elevation might have different AP visibility
            wifi_count = max(1, np.random.poisson(3 + elevation))
            wifi_avg_strength = -45 - elevation * 2 + np.random.normal(0, 5)
            wifi_max_strength = wifi_avg_strength + np.random.uniform(5, 15)
            wifi_std_strength = np.random.uniform(2, 8)
            
            # Bluetooth: fewer beacons at higher elevations
            bt_count = max(0, np.random.poisson(max(1, 3 - elevation)))
            bt_avg_strength = -65 + np.random.normal(0, 10) if bt_count > 0 else -100
            bt_max_strength = bt_avg_strength + np.random.uniform(0, 10) if bt_count > 0 else -100
            
            # Motion data (less relevant for elevation but included)
            accel_magnitude = 9.81 + np.random.normal(0, 0.5)
            gyro_magnitude = np.random.exponential(0.1)
            
            # Magnetometer (can be affected by building structure)
            mag_x = 20 + elevation * 0.5 + np.random.normal(0, 3)
            mag_y = np.random.normal(0, 5)
            mag_z = -45 + elevation * 0.3 + np.random.normal(0, 4)
            
            features = {
                'gps_lat': lat,
                'gps_lon': lon,
                'gps_accuracy': np.random.uniform(3, 15),
                'barometer_pressure': pressure,
                'wifi_count': wifi_count,
                'wifi_max_strength': wifi_max_strength,
                'wifi_avg_strength': wifi_avg_strength,
                'wifi_std_strength': wifi_std_strength,
                'bt_count': bt_count,
                'bt_max_strength': bt_max_strength,
                'bt_avg_strength': bt_avg_strength,
                'accel_magnitude': accel_magnitude,
                'gyro_magnitude': gyro_magnitude,
                'magnetometer_x': mag_x,
                'magnetometer_y': mag_y,
                'magnetometer_z': mag_z
            }
            
            features_list.append(features)
            labels_list.append(elevation)
        
        features_df = pd.DataFrame(features_list)
        labels_series = pd.Series(labels_list)
        
        self.feature_names = features_df.columns.tolist()
        
        return features_df, labels_series
    
    def train_model(self, features: pd.DataFrame, labels: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train XGBoost model for elevation prediction.
        
        Args:
            features: Feature dataframe
            labels: Target labels (elevation levels)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            objective='multi:softprob',
            eval_metric='mlogloss'
        )
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring='accuracy'
        )
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': dict(zip(
                self.feature_names, 
                self.model.feature_importances_
            ))
        }
    
    def predict_elevation(self, sensor_features: Dict) -> Tuple[int, float]:
        """
        Predict elevation level from sensor features.
        
        Args:
            sensor_features: Dictionary of sensor features
            
        Returns:
            Tuple of (predicted_elevation, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure features are in correct order
        feature_vector = np.array([
            sensor_features.get(feature, 0) for feature in self.feature_names
        ]).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Get prediction and probability
        prediction = self.model.predict(feature_vector_scaled)[0]
        probabilities = self.model.predict_proba(feature_vector_scaled)[0]
        confidence = max(probabilities)
        
        return int(prediction), float(confidence)
    
    def predict_batch(self, sensor_features_list: List[Dict]) -> List[Tuple[int, float]]:
        """
        Predict elevation levels for multiple sensor readings.
        
        Args:
            sensor_features_list: List of sensor feature dictionaries
            
        Returns:
            List of (predicted_elevation, confidence) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare feature matrix
        feature_matrix = []
        for sensor_features in sensor_features_list:
            feature_vector = [
                sensor_features.get(feature, 0) for feature in self.feature_names
            ]
            feature_matrix.append(feature_vector)
        
        feature_matrix = np.array(feature_matrix)
        feature_matrix_scaled = self.scaler.transform(feature_matrix)
        
        # Get predictions and probabilities
        predictions = self.model.predict(feature_matrix_scaled)
        probabilities = self.model.predict_proba(feature_matrix_scaled)
        confidences = np.max(probabilities, axis=1)
        
        return [(int(pred), float(conf)) for pred, conf in zip(predictions, confidences)]
    
    def save_model(self, model_path: str):
        """Save trained model and scaler to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
    
    def load_model(self, model_path: str):
        """Load trained model and scaler from disk."""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        return dict(zip(self.feature_names, self.model.feature_importances_))


class ElevationTracker:
    """Tracks elevation predictions over time to improve accuracy."""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.prediction_history = []
    
    def add_prediction(self, elevation: int, confidence: float, timestamp: datetime = None):
        """Add a new elevation prediction to the tracking window."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.prediction_history.append({
            'elevation': elevation,
            'confidence': confidence,
            'timestamp': timestamp
        })
        
        # Keep only the most recent predictions
        if len(self.prediction_history) > self.window_size:
            self.prediction_history.pop(0)
    
    def get_smoothed_elevation(self) -> Tuple[int, float]:
        """
        Get smoothed elevation prediction based on recent history.
        
        Returns:
            Tuple of (smoothed_elevation, average_confidence)
        """
        if not self.prediction_history:
            return 0, 0.0
        
        # Weight predictions by confidence and recency
        total_weight = 0
        weighted_sum = 0
        confidence_sum = 0
        
        for i, pred in enumerate(self.prediction_history):
            # Recent predictions get higher weight
            time_weight = (i + 1) / len(self.prediction_history)
            # Higher confidence predictions get higher weight
            conf_weight = pred['confidence']
            
            weight = time_weight * conf_weight
            weighted_sum += pred['elevation'] * weight
            total_weight += weight
            confidence_sum += pred['confidence']
        
        smoothed_elevation = int(round(weighted_sum / total_weight))
        avg_confidence = confidence_sum / len(self.prediction_history)
        
        return smoothed_elevation, avg_confidence


if __name__ == "__main__":
    # Example usage and testing
    print("Training XGBoost elevation prediction model...")
    
    predictor = ElevationPredictor()
    
    # Create synthetic training data
    features, labels = predictor.create_synthetic_training_data(n_samples=1000)
    
    # Train the model
    metrics = predictor.train_model(features, labels)
    
    print(f"Model Accuracy: {metrics['accuracy']:.3f}")
    print(f"CV Mean Accuracy: {metrics['cv_mean_accuracy']:.3f} ± {metrics['cv_std_accuracy']:.3f}")
    
    print("\nFeature Importance:")
    for feature, importance in sorted(metrics['feature_importance'].items(), 
                                    key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.3f}")
    
    # Test prediction
    test_features = {
        'gps_lat': 29.5630,
        'gps_lon': 106.5516,
        'gps_accuracy': 5.0,
        'barometer_pressure': 1010.0,  # Indicates ~3rd floor
        'wifi_count': 4,
        'wifi_max_strength': -35,
        'wifi_avg_strength': -45,
        'wifi_std_strength': 5,
        'bt_count': 2,
        'bt_max_strength': -55,
        'bt_avg_strength': -65,
        'accel_magnitude': 9.8,
        'gyro_magnitude': 0.1,
        'magnetometer_x': 22,
        'magnetometer_y': 1,
        'magnetometer_z': -44
    }
    
    elevation, confidence = predictor.predict_elevation(test_features)
    print(f"\nTest Prediction: Elevation Level {elevation} (Confidence: {confidence:.3f})")
    
    # Save model
    predictor.save_model("models/elevation_predictor.joblib")
    print("\nModel saved successfully!")