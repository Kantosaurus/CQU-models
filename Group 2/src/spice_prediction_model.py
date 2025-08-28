"""
XGBoost Spice Prediction Model
Group 2 - Chongqing Food Spice Prediction

This module implements the core XGBoost model for personalized spice level prediction:
1. Feature engineering from dish and user data
2. XGBoost model training and optimization
3. Personalized prediction based on user tolerance profiles
4. Reviewer bias correction and calibration
5. Model evaluation and interpretability
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from data_collection import SpiceDataCollector
from user_profiling import UserProfilingSystem

logger = logging.getLogger(__name__)

class SpicePredictionModel:
    """XGBoost-based personalized spice level prediction model"""
    
    def __init__(self, db_path: str = "data/spice_database.db", 
                 model_path: str = "models/"):
        self.db_path = db_path
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        
        # Initialize data collector and profiler
        self.data_collector = SpiceDataCollector(db_path)
        self.user_profiler = UserProfilingSystem(db_path)
        
        # Model components
        self.xgb_model = None
        self.feature_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Model parameters
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with comprehensive feature engineering"""
        logger.info("Preparing training data...")
        
        # Get raw training data
        raw_data = self.data_collector.export_training_data()
        
        if raw_data.empty:
            raise ValueError("No training data available. Run data collection first.")
        
        # Feature engineering
        features_df = self._engineer_features(raw_data)
        target = raw_data['target_spice_level']
        
        logger.info(f"Prepared {len(features_df)} training samples with {len(features_df.columns)} features")
        
        return features_df, target
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive feature engineering for spice prediction"""
        features = pd.DataFrame()
        
        # === User Features ===
        features['user_tolerance_level'] = data['user_tolerance_level'].fillna(3)
        
        # User bias features (calculated dynamically)
        features['user_bias'] = data.groupby('user_id')['target_spice_level'].transform(
            lambda x: x.mean() - data['base_spice_level'].mean()
        )
        
        # User rating patterns
        features['user_avg_rating'] = data.groupby('user_id')['overall_rating'].transform('mean')
        features['user_rating_std'] = data.groupby('user_id')['target_spice_level'].transform('std').fillna(1.0)
        features['user_experience'] = data.groupby('user_id').cumcount() + 1  # Number of ratings so far
        
        # === Dish Features ===
        features['base_spice_level'] = data['base_spice_level'].fillna(3)
        features['lab_scoville'] = data['lab_scoville'].fillna(0)
        features['log_scoville'] = np.log1p(features['lab_scoville'])  # Log transform for better distribution
        features['price'] = data['price'].fillna(data['price'].median())
        features['price_per_spice'] = features['price'] / (features['base_spice_level'] + 1)
        
        # === Ingredient Features ===
        # Count of spice ingredients
        spice_ingredient_counts = data['spice_ingredients'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        features['spice_ingredient_count'] = spice_ingredient_counts
        
        # Specific Chongqing spice ingredients (binary features)
        chongqing_spices = ['tian_jiao', 'xiao_mi_la', 'er_jin_tiao', 'dou_ban_jiang', 
                          'hua_jiao', 'gan_la_jiao', 'chao_tian_jiao']
        
        for spice in chongqing_spices:
            features[f'has_{spice}'] = data['spice_ingredients'].apply(
                lambda x: 1 if isinstance(x, list) and any(spice in ingredient for ingredient in x) else 0
            )
        
        # Main ingredient categories
        main_ingredients_text = data['main_ingredients'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else ''
        )
        
        ingredient_categories = ['meat', 'fish', 'vegetable', 'tofu', 'noodles', 'rice']
        for category in ingredient_categories:
            features[f'has_{category}'] = main_ingredients_text.str.contains(category, case=False, na=False).astype(int)
        
        # === Categorical Features ===
        # Cuisine type encoding
        le_cuisine = LabelEncoder()
        features['cuisine_type_encoded'] = le_cuisine.fit_transform(data['cuisine_type'].fillna('Unknown'))
        self.feature_encoders['cuisine_type'] = le_cuisine
        
        # Cooking method encoding with intensity
        cooking_intensity_map = {
            'stir_fry': 1.0, 'deep_fry': 0.8, 'boil': 0.7, 'steam': 0.6,
            'grill': 1.2, 'hot_pot': 1.5, 'dry_pot': 1.8
        }
        features['cooking_intensity'] = data['cooking_method'].map(cooking_intensity_map).fillna(1.0)
        
        le_cooking = LabelEncoder()
        features['cooking_method_encoded'] = le_cooking.fit_transform(data['cooking_method'].fillna('stir_fry'))
        self.feature_encoders['cooking_method'] = le_cooking
        
        # === Interaction Features ===
        # User tolerance vs dish spice interaction
        features['tolerance_spice_diff'] = features['user_tolerance_level'] - features['base_spice_level']
        features['tolerance_spice_ratio'] = features['user_tolerance_level'] / (features['base_spice_level'] + 1)
        
        # Price-spice value perception
        features['spice_value'] = features['base_spice_level'] / (features['price'] + 1)
        
        # Restaurant-level features
        features['restaurant_avg_spice'] = data.groupby('restaurant_id')['base_spice_level'].transform('mean')
        features['restaurant_spice_variance'] = data.groupby('restaurant_id')['base_spice_level'].transform('std').fillna(0)
        
        # === Time-based Features ===
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            features['hour_of_day'] = data['timestamp'].dt.hour
            features['day_of_week'] = data['timestamp'].dt.dayofweek
            features['is_weekend'] = (data['timestamp'].dt.dayofweek >= 5).astype(int)
        
        # === Advanced User Modeling ===
        # User's historical spice preference trend
        features['user_spice_trend'] = data.groupby('user_id')['target_spice_level'].transform(
            lambda x: self._calculate_trend(x.values) if len(x) > 2 else 0
        )
        
        # User's consistency in rating
        features['user_rating_consistency'] = 1 / (features['user_rating_std'] + 0.1)
        
        # Store feature names for later use
        self.feature_names = list(features.columns)
        
        return features
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate trend in user's spice ratings over time"""
        if len(values) < 3:
            return 0.0
        
        x = np.arange(len(values))
        try:
            slope = np.polyfit(x, values, 1)[0]
            return slope
        except:
            return 0.0
    
    def train_model(self, features: pd.DataFrame, target: pd.Series, 
                   validation_split: float = 0.2) -> Dict[str, Any]:
        """Train XGBoost model with hyperparameter optimization"""
        logger.info("Training XGBoost model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, target, test_size=validation_split, random_state=42, stratify=target
        )
        
        # Scale numerical features
        numerical_features = features.select_dtypes(include=[np.number]).columns
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        
        X_train_scaled[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
        X_val_scaled[numerical_features] = self.scaler.transform(X_val[numerical_features])
        
        # Hyperparameter optimization
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        xgb_regressor = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            xgb_regressor, param_grid, 
            cv=5, scoring='neg_mean_absolute_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        self.xgb_model = grid_search.best_estimator_
        
        # Validation predictions
        y_val_pred = self.xgb_model.predict(X_val_scaled)
        
        # Calculate metrics
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        # Classification accuracy (within ±0.5 levels)
        val_accuracy = np.mean(np.abs(y_val - y_val_pred) <= 0.5)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        training_results = {
            'best_params': grid_search.best_params_,
            'validation_mae': val_mae,
            'validation_rmse': val_rmse,
            'validation_accuracy': val_accuracy,
            'feature_importance': feature_importance,
            'cv_scores': -grid_search.cv_results_['mean_test_score']
        }
        
        logger.info(f"Model training completed. Validation MAE: {val_mae:.3f}, Accuracy: {val_accuracy:.3f}")
        
        # Save model
        self.save_model()
        
        return training_results
    
    def predict_personalized_spice_level(self, user_id: str, dish_id: str) -> Dict[str, Any]:
        """Predict personalized spice level for a user-dish pair"""
        if self.xgb_model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Get user profile
        user_profile = self.user_profiler.get_user_profile(user_id)
        if not user_profile:
            # Build profile for new user
            user_profile = self.user_profiler.build_complete_profile(user_id)
        
        # Get dish data
        dish_data = self.data_collector.get_dish_data(dish_id)
        if not dish_data:
            raise ValueError(f"Dish {dish_id} not found")
        
        # Create feature vector
        features = self._create_prediction_features(user_profile, dish_data)
        
        # Make prediction
        raw_prediction = self.xgb_model.predict(features)[0]
        
        # Apply bias correction
        bias_corrected_prediction = raw_prediction - user_profile.rating_bias
        
        # Apply tolerance-based adjustment
        tolerance_adjusted_prediction = self._apply_tolerance_adjustment(
            bias_corrected_prediction, user_profile, dish_data
        )
        
        # Ensure prediction is within valid range (1-5)
        final_prediction = max(1, min(5, round(tolerance_adjusted_prediction, 1)))
        
        # Calculate confidence based on user profile confidence and model certainty
        prediction_confidence = self._calculate_prediction_confidence(
            user_profile, dish_data, features
        )
        
        # Generate explanation
        explanation = self._generate_prediction_explanation(
            user_profile, dish_data, final_prediction, features
        )
        
        result = {
            'predicted_spice_level': final_prediction,
            'confidence': prediction_confidence,
            'explanation': explanation,
            'raw_model_output': raw_prediction,
            'bias_correction': user_profile.rating_bias,
            'user_tolerance_level': user_profile.tolerance_level,
            'dish_base_spice': dish_data.base_spice_level
        }
        
        return result
    
    def _create_prediction_features(self, user_profile, dish_data) -> pd.DataFrame:
        """Create feature vector for prediction"""
        features = pd.DataFrame(index=[0])
        
        # User features
        features['user_tolerance_level'] = user_profile.tolerance_level
        features['user_bias'] = user_profile.rating_bias
        features['user_avg_rating'] = 4.0  # Default, would be calculated from history
        features['user_rating_std'] = 1.0
        features['user_experience'] = 10  # Default
        
        # Dish features
        features['base_spice_level'] = dish_data.base_spice_level or 3
        features['lab_scoville'] = dish_data.lab_scoville or 0
        features['log_scoville'] = np.log1p(features['lab_scoville'].iloc[0])
        features['price'] = dish_data.price or 30.0
        features['price_per_spice'] = features['price'] / (features['base_spice_level'] + 1)
        
        # Spice ingredients
        features['spice_ingredient_count'] = len(dish_data.spice_ingredients)
        
        chongqing_spices = ['tian_jiao', 'xiao_mi_la', 'er_jin_tiao', 'dou_ban_jiang', 
                          'hua_jiao', 'gan_la_jiao', 'chao_tian_jiao']
        
        for spice in chongqing_spices:
            features[f'has_{spice}'] = 1 if any(spice in ingredient for ingredient in dish_data.spice_ingredients) else 0
        
        # Main ingredients
        main_ingredients_text = ' '.join(dish_data.main_ingredients)
        ingredient_categories = ['meat', 'fish', 'vegetable', 'tofu', 'noodles', 'rice']
        for category in ingredient_categories:
            features[f'has_{category}'] = 1 if category in main_ingredients_text.lower() else 0
        
        # Categorical features
        if 'cuisine_type' in self.feature_encoders:
            try:
                features['cuisine_type_encoded'] = self.feature_encoders['cuisine_type'].transform([dish_data.cuisine_type])[0]
            except:
                features['cuisine_type_encoded'] = 0  # Unknown category
        else:
            features['cuisine_type_encoded'] = 0
        
        # Cooking method
        cooking_intensity_map = {
            'stir_fry': 1.0, 'deep_fry': 0.8, 'boil': 0.7, 'steam': 0.6,
            'grill': 1.2, 'hot_pot': 1.5, 'dry_pot': 1.8
        }
        features['cooking_intensity'] = cooking_intensity_map.get(dish_data.cooking_method, 1.0)
        
        if 'cooking_method' in self.feature_encoders:
            try:
                features['cooking_method_encoded'] = self.feature_encoders['cooking_method'].transform([dish_data.cooking_method])[0]
            except:
                features['cooking_method_encoded'] = 0
        else:
            features['cooking_method_encoded'] = 0
        
        # Interaction features
        features['tolerance_spice_diff'] = features['user_tolerance_level'] - features['base_spice_level']
        features['tolerance_spice_ratio'] = features['user_tolerance_level'] / (features['base_spice_level'] + 1)
        features['spice_value'] = features['base_spice_level'] / (features['price'] + 1)
        
        # Default values for restaurant and time features
        features['restaurant_avg_spice'] = features['base_spice_level']
        features['restaurant_spice_variance'] = 1.0
        features['hour_of_day'] = 12
        features['day_of_week'] = 3
        features['is_weekend'] = 0
        features['user_spice_trend'] = user_profile.tolerance_trend
        features['user_rating_consistency'] = 0.8
        
        # Ensure all expected features are present
        for feature_name in self.feature_names:
            if feature_name not in features.columns:
                features[feature_name] = 0
        
        # Reorder columns to match training data
        features = features[self.feature_names]
        
        # Scale numerical features
        numerical_features = features.select_dtypes(include=[np.number]).columns
        features[numerical_features] = self.scaler.transform(features[numerical_features])
        
        return features
    
    def _apply_tolerance_adjustment(self, prediction: float, user_profile, dish_data) -> float:
        """Apply user tolerance-based adjustment to prediction"""
        # If dish spice level is within user's preferred range, reduce predicted spice level
        dish_spice = dish_data.base_spice_level or 3
        min_pref, max_pref = user_profile.preferred_spice_range
        
        if min_pref <= dish_spice <= max_pref:
            # User likes this spice level, so they might rate it lower
            adjustment = -0.3
        elif dish_spice < min_pref:
            # Dish is too mild for user's preference
            adjustment = 0.2
        else:
            # Dish is spicier than user prefers
            adjustment = 0.5
        
        return prediction + adjustment
    
    def _calculate_prediction_confidence(self, user_profile, dish_data, features) -> float:
        """Calculate confidence in the prediction"""
        # Base confidence from user profile
        base_confidence = user_profile.confidence_score
        
        # Model uncertainty (could be enhanced with ensemble methods)
        model_confidence = 0.8  # Would calculate from prediction variance in ensemble
        
        # Data completeness confidence
        completeness_factors = [
            1.0 if dish_data.lab_scoville else 0.7,
            1.0 if dish_data.base_spice_level else 0.8,
            1.0 if len(dish_data.spice_ingredients) > 0 else 0.9
        ]
        completeness_confidence = np.mean(completeness_factors)
        
        # Combined confidence
        overall_confidence = (base_confidence * 0.4 + 
                            model_confidence * 0.4 + 
                            completeness_confidence * 0.2)
        
        return min(0.95, max(0.3, overall_confidence))
    
    def _generate_prediction_explanation(self, user_profile, dish_data, 
                                       prediction: float, features: pd.DataFrame) -> str:
        """Generate human-readable explanation for the prediction"""
        explanations = []
        
        # User tolerance context
        tolerance_levels = {1: "very low", 2: "low", 3: "moderate", 4: "high", 5: "very high"}
        user_tolerance_desc = tolerance_levels.get(user_profile.tolerance_level, "moderate")
        explanations.append(f"Based on your {user_tolerance_desc} spice tolerance")
        
        # Dish characteristics
        dish_spice = dish_data.base_spice_level or 3
        if dish_spice >= 4:
            explanations.append(f"this dish is quite spicy (level {dish_spice})")
        elif dish_spice <= 2:
            explanations.append(f"this dish is mild (level {dish_spice})")
        else:
            explanations.append(f"this dish has moderate spice (level {dish_spice})")
        
        # Spice ingredients impact
        if any('chao_tian_jiao' in ing or 'xiao_mi_la' in ing for ing in dish_data.spice_ingredients):
            explanations.append("contains very hot Chongqing peppers")
        elif any('hua_jiao' in ing for ing in dish_data.spice_ingredients):
            explanations.append("includes numbing Sichuan peppercorns")
        
        # Cooking method impact
        if dish_data.cooking_method == 'hot_pot':
            explanations.append("hot pot cooking intensifies the spice")
        elif dish_data.cooking_method == 'dry_pot':
            explanations.append("dry pot style makes it very spicy")
        
        # Personal bias adjustment
        if abs(user_profile.rating_bias) > 0.3:
            if user_profile.rating_bias > 0:
                explanations.append("adjusted up since you typically rate dishes as less spicy")
            else:
                explanations.append("adjusted down since you typically rate dishes as more spicy")
        
        return "; ".join(explanations) + f". Predicted spice level for you: {prediction}"
    
    def save_model(self, model_name: str = "spice_prediction_xgboost.joblib"):
        """Save trained model and encoders"""
        model_data = {
            'xgb_model': self.xgb_model,
            'feature_encoders': self.feature_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_params': self.xgb_params
        }
        
        model_file = self.model_path / model_name
        joblib.dump(model_data, model_file)
        logger.info(f"Model saved to {model_file}")
    
    def load_model(self, model_name: str = "spice_prediction_xgboost.joblib"):
        """Load pre-trained model and encoders"""
        model_file = self.model_path / model_name
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file {model_file} not found")
        
        model_data = joblib.load(model_file)
        
        self.xgb_model = model_data['xgb_model']
        self.feature_encoders = model_data['feature_encoders']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from {model_file}")
    
    def evaluate_model(self, test_features: pd.DataFrame, test_target: pd.Series) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        if self.xgb_model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Scale features
        numerical_features = test_features.select_dtypes(include=[np.number]).columns
        test_features_scaled = test_features.copy()
        test_features_scaled[numerical_features] = self.scaler.transform(test_features[numerical_features])
        
        # Predictions
        predictions = self.xgb_model.predict(test_features_scaled)
        
        # Regression metrics
        mae = mean_absolute_error(test_target, predictions)
        rmse = np.sqrt(mean_squared_error(test_target, predictions))
        
        # Classification accuracy (within ±0.5 levels)
        accuracy_05 = np.mean(np.abs(test_target - predictions) <= 0.5)
        accuracy_1 = np.mean(np.abs(test_target - predictions) <= 1.0)
        
        # Per-spice-level analysis
        spice_level_analysis = {}
        for level in range(1, 6):
            level_mask = test_target == level
            if level_mask.sum() > 0:
                level_predictions = predictions[level_mask]
                level_mae = mean_absolute_error([level] * level_mask.sum(), level_predictions)
                spice_level_analysis[f'level_{level}_mae'] = level_mae
        
        evaluation_results = {
            'mae': mae,
            'rmse': rmse,
            'accuracy_within_0.5': accuracy_05,
            'accuracy_within_1.0': accuracy_1,
            'spice_level_analysis': spice_level_analysis,
            'predictions_sample': list(zip(test_target[:10].values, predictions[:10]))
        }
        
        return evaluation_results
    
    def plot_feature_importance(self, top_n: int = 20):
        """Plot feature importance"""
        if self.xgb_model is None:
            raise ValueError("Model not trained")
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance, y='feature', x='importance')
        plt.title('Top Feature Importances for Spice Level Prediction')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
        return feature_importance

if __name__ == "__main__":
    # Example usage
    model = SpicePredictionModel()
    
    # Initialize sample data if needed
    model.data_collector.initialize_sample_data()
    
    # Prepare training data
    features, target = model.prepare_training_data()
    
    # Train model
    results = model.train_model(features, target)
    print(f"Training completed. Validation MAE: {results['validation_mae']:.3f}")
    
    # Make a prediction
    prediction = model.predict_personalized_spice_level("U001", "CQ002")
    print(f"\nPrediction for user U001 and dish CQ002:")
    print(f"Predicted spice level: {prediction['predicted_spice_level']}")
    print(f"Confidence: {prediction['confidence']:.2f}")
    print(f"Explanation: {prediction['explanation']}")