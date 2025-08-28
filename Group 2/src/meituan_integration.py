"""
Meituan API Integration Layer
Group 2 - Chongqing Food Spice Prediction

This module provides integration with Meituan platform:
1. API endpoints for spice level predictions
2. Restaurant and dish data synchronization
3. User rating collection and processing
4. Real-time spice level display integration
5. Recommendation delivery system
"""

import requests
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import sqlite3
from dataclasses import dataclass, asdict
import logging
from flask import Flask, request, jsonify
import pandas as pd
from functools import wraps
import hashlib
import hmac

from spice_prediction_model import SpicePredictionModel
from user_profiling import UserProfilingSystem
from bias_correction import ReviewerBiasCorrector
from tolerance_building import ToleranceBuildingEngine
from data_collection import SpiceDataCollector, DishData, UserRating

logger = logging.getLogger(__name__)

@dataclass
class MeituanDish:
    """Meituan dish data structure"""
    meituan_dish_id: str
    restaurant_id: str
    name: str
    price: float
    category: str
    description: str
    ingredients: List[str]
    image_url: str
    availability: bool
    
@dataclass
class SpicePredictionResponse:
    """Response structure for spice predictions"""
    dish_id: str
    predicted_spice_level: float
    confidence: float
    explanation: str
    user_recommendation: str
    tolerance_building_suggestion: Optional[str] = None

class MeituanSpiceAPI:
    """Flask API for Meituan integration"""
    
    def __init__(self, db_path: str = "data/spice_database.db"):
        self.app = Flask(__name__)
        self.db_path = db_path
        
        # Initialize AI components
        self.prediction_model = SpicePredictionModel(db_path)
        self.user_profiler = UserProfilingSystem(db_path)
        self.bias_corrector = ReviewerBiasCorrector(db_path)
        self.tolerance_engine = ToleranceBuildingEngine(db_path)
        self.data_collector = SpiceDataCollector(db_path)
        
        # API configuration
        self.api_key = "meituan_spice_api_key_2024"  # In production, use environment variable
        self.rate_limit = 1000  # requests per minute
        self.cache_duration = 300  # 5 minutes cache
        
        # Setup API endpoints
        self.setup_routes()
        
        # Load model if available
        try:
            self.prediction_model.load_model()
            logger.info("Prediction model loaded successfully")
        except FileNotFoundError:
            logger.warning("No trained model found. Training with sample data...")
            self._initialize_and_train_model()
    
    def _initialize_and_train_model(self):
        """Initialize sample data and train model"""
        try:
            # Initialize sample data
            self.data_collector.initialize_sample_data()
            
            # Train model
            features, target = self.prediction_model.prepare_training_data()
            self.prediction_model.train_model(features, target)
            
            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"Failed to initialize and train model: {e}")
    
    def setup_routes(self):
        """Setup Flask API routes"""
        
        @self.app.route('/api/v1/health', methods=['GET'])
        def health_check():
            """API health check"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'services': {
                    'prediction_model': self.prediction_model.xgb_model is not None,
                    'user_profiling': True,
                    'bias_correction': True,
                    'tolerance_building': True
                }
            })
        
        @self.app.route('/api/v1/predict_spice', methods=['POST'])
        @self.require_api_key
        @self.rate_limit_decorator
        def predict_spice_level():
            """Predict personalized spice level for user-dish pair"""
            try:
                data = request.get_json()
                
                # Validate request
                required_fields = ['user_id', 'dish_id']
                if not all(field in data for field in required_fields):
                    return jsonify({'error': 'Missing required fields'}), 400
                
                user_id = data['user_id']
                dish_id = data['dish_id']
                
                # Make prediction
                prediction = self.prediction_model.predict_personalized_spice_level(user_id, dish_id)
                
                # Generate user recommendation
                recommendation = self._generate_user_recommendation(prediction, user_id)
                
                # Check tolerance building opportunity
                tolerance_suggestion = self._get_tolerance_building_suggestion(user_id, dish_id, prediction)
                
                response = SpicePredictionResponse(
                    dish_id=dish_id,
                    predicted_spice_level=prediction['predicted_spice_level'],
                    confidence=prediction['confidence'],
                    explanation=prediction['explanation'],
                    user_recommendation=recommendation,
                    tolerance_building_suggestion=tolerance_suggestion
                )
                
                return jsonify(asdict(response))
                
            except Exception as e:
                logger.error(f"Error in predict_spice_level: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/v1/user/profile', methods=['GET', 'POST'])
        @self.require_api_key
        def user_profile():
            """Get or update user tolerance profile"""
            try:
                if request.method == 'GET':
                    user_id = request.args.get('user_id')
                    if not user_id:
                        return jsonify({'error': 'user_id required'}), 400
                    
                    profile = self.user_profiler.get_user_profile(user_id)
                    if not profile:
                        profile = self.user_profiler.build_complete_profile(user_id)
                    
                    return jsonify({
                        'user_id': profile.user_id,
                        'tolerance_level': profile.tolerance_level,
                        'confidence': profile.confidence_score,
                        'preferred_range': profile.preferred_spice_range,
                        'cuisine_preferences': profile.cuisine_preferences,
                        'last_updated': profile.last_updated.isoformat()
                    })
                
                elif request.method == 'POST':
                    data = request.get_json()
                    user_id = data.get('user_id')
                    dish_id = data.get('dish_id')
                    spice_rating = data.get('spice_rating')
                    overall_rating = data.get('overall_rating')
                    
                    if not all([user_id, dish_id, spice_rating]):
                        return jsonify({'error': 'Missing required fields'}), 400
                    
                    # Update user profile with new rating
                    self.user_profiler.update_profile_with_new_rating(
                        user_id, dish_id, spice_rating, overall_rating or 4.0
                    )
                    
                    return jsonify({'status': 'profile updated'})
                    
            except Exception as e:
                logger.error(f"Error in user_profile: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/v1/rating/submit', methods=['POST'])
        @self.require_api_key
        def submit_rating():
            """Submit new user rating with bias correction"""
            try:
                data = request.get_json()
                
                required_fields = ['user_id', 'dish_id', 'spice_rating', 'overall_rating']
                if not all(field in data for field in required_fields):
                    return jsonify({'error': 'Missing required fields'}), 400
                
                # Apply bias correction to the rating
                corrected_rating = self.bias_corrector.apply_bias_correction(
                    data['user_id'], data['spice_rating'], data['dish_id']
                )
                
                # Store original and corrected rating
                rating = UserRating(
                    user_id=data['user_id'],
                    dish_id=data['dish_id'],
                    spice_rating=data['spice_rating'],
                    overall_rating=data['overall_rating'],
                    timestamp=datetime.now(),
                    review_text=data.get('review_text'),
                    user_tolerance_level=data.get('user_tolerance_level')
                )
                
                self.data_collector.add_user_rating(rating)
                
                # Update user profile
                self.user_profiler.update_profile_with_new_rating(
                    data['user_id'], data['dish_id'], 
                    data['spice_rating'], data['overall_rating']
                )
                
                return jsonify({
                    'status': 'rating submitted',
                    'original_rating': corrected_rating.original_rating,
                    'corrected_rating': corrected_rating.corrected_rating,
                    'correction_applied': corrected_rating.correction_applied,
                    'bias_explanation': corrected_rating.correction_reason
                })
                
            except Exception as e:
                logger.error(f"Error in submit_rating: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/v1/tolerance/plan', methods=['GET', 'POST'])
        @self.require_api_key
        def tolerance_building():
            """Get or create tolerance building plan"""
            try:
                if request.method == 'GET':
                    user_id = request.args.get('user_id')
                    if not user_id:
                        return jsonify({'error': 'user_id required'}), 400
                    
                    plan = self.tolerance_engine.get_active_plan(user_id)
                    if not plan:
                        return jsonify({'message': 'No active plan found'}), 404
                    
                    return jsonify({
                        'user_id': plan.user_id,
                        'current_tolerance': plan.current_tolerance,
                        'target_tolerance': plan.target_tolerance,
                        'timeline_weeks': plan.estimated_timeline_weeks,
                        'weekly_goals': plan.weekly_goals,
                        'safety_guidelines': plan.safety_guidelines,
                        'progress_milestones': plan.progress_milestones
                    })
                
                elif request.method == 'POST':
                    data = request.get_json()
                    user_id = data.get('user_id')
                    target_tolerance = data.get('target_tolerance')
                    
                    if not user_id:
                        return jsonify({'error': 'user_id required'}), 400
                    
                    plan = self.tolerance_engine.create_tolerance_building_plan(
                        user_id, target_tolerance
                    )
                    
                    return jsonify({
                        'status': 'plan created',
                        'plan_id': plan.user_id,
                        'timeline_weeks': plan.estimated_timeline_weeks,
                        'target_tolerance': plan.target_tolerance
                    })
                    
            except Exception as e:
                logger.error(f"Error in tolerance_building: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/v1/dishes/sync', methods=['POST'])
        @self.require_api_key
        def sync_dishes():
            """Sync dish data from Meituan"""
            try:
                data = request.get_json()
                
                if 'dishes' not in data:
                    return jsonify({'error': 'dishes data required'}), 400
                
                synced_count = 0
                for dish_data in data['dishes']:
                    # Convert Meituan format to internal format
                    dish = self._convert_meituan_dish(dish_data)
                    if dish:
                        self.data_collector.add_dish(dish)
                        synced_count += 1
                
                return jsonify({
                    'status': 'sync completed',
                    'synced_dishes': synced_count
                })
                
            except Exception as e:
                logger.error(f"Error in sync_dishes: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/v1/restaurants/<restaurant_id>/dishes', methods=['GET'])
        @self.require_api_key
        def get_restaurant_dishes(restaurant_id):
            """Get dishes for a restaurant with spice predictions"""
            try:
                user_id = request.args.get('user_id')
                
                # Get dishes for restaurant
                dishes = self._get_restaurant_dishes(restaurant_id)
                
                if not dishes:
                    return jsonify({'dishes': []})
                
                # Add spice predictions if user_id provided
                if user_id:
                    for dish in dishes:
                        try:
                            prediction = self.prediction_model.predict_personalized_spice_level(
                                user_id, dish['dish_id']
                            )
                            dish['predicted_spice_level'] = prediction['predicted_spice_level']
                            dish['prediction_confidence'] = prediction['confidence']
                            dish['spice_explanation'] = prediction['explanation']
                        except:
                            dish['predicted_spice_level'] = dish.get('base_spice_level', 3)
                            dish['prediction_confidence'] = 0.5
                            dish['spice_explanation'] = 'Default spice level'
                
                return jsonify({'dishes': dishes})
                
            except Exception as e:
                logger.error(f"Error in get_restaurant_dishes: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/v1/analytics/bias', methods=['GET'])
        @self.require_api_key
        def bias_analytics():
            """Get bias correction analytics"""
            try:
                analytics = self.bias_corrector.analyze_bias_patterns()
                correction_stats = self.bias_corrector.get_correction_statistics()
                
                return jsonify({
                    'bias_patterns': analytics,
                    'correction_statistics': correction_stats
                })
                
            except Exception as e:
                logger.error(f"Error in bias_analytics: {e}")
                return jsonify({'error': 'Internal server error'}), 500
    
    def require_api_key(self, f):
        """Decorator to require API key authentication"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')
            if not api_key or not self._validate_api_key(api_key):
                return jsonify({'error': 'Invalid API key'}), 401
            return f(*args, **kwargs)
        return decorated_function
    
    def rate_limit_decorator(self, f):
        """Simple rate limiting decorator"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # In production, implement proper rate limiting with Redis
            return f(*args, **kwargs)
        return decorated_function
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        return api_key == self.api_key
    
    def _generate_user_recommendation(self, prediction: Dict[str, Any], user_id: str) -> str:
        """Generate user-friendly recommendation"""
        spice_level = prediction['predicted_spice_level']
        confidence = prediction['confidence']
        
        # Get user profile for context
        profile = self.user_profiler.get_user_profile(user_id)
        
        recommendations = []
        
        if spice_level <= 2:
            recommendations.append("🟢 This dish should be mild for you")
        elif spice_level <= 3.5:
            recommendations.append("🟡 This dish has moderate spice - perfect for most people")
        elif spice_level <= 4.5:
            recommendations.append("🟠 This dish is quite spicy - proceed with caution")
        else:
            recommendations.append("🔴 This is very spicy - only for spice lovers!")
        
        if confidence < 0.6:
            recommendations.append("(Low confidence - we need more data about your preferences)")
        elif confidence > 0.8:
            recommendations.append("(High confidence - based on your taste profile)")
        
        if profile and spice_level > profile.preferred_spice_range[1]:
            recommendations.append("This might be spicier than your usual preference")
        
        return " ".join(recommendations)
    
    def _get_tolerance_building_suggestion(self, user_id: str, dish_id: str, 
                                         prediction: Dict[str, Any]) -> Optional[str]:
        """Get tolerance building suggestion if applicable"""
        try:
            # Check if user has active tolerance building plan
            plan = self.tolerance_engine.get_active_plan(user_id)
            if not plan:
                return None
            
            predicted_level = prediction['predicted_spice_level']
            target_level = plan.target_tolerance
            
            if predicted_level > target_level:
                return f"This dish (level {predicted_level:.1f}) would help you progress toward your target tolerance of {target_level}"
            elif predicted_level == target_level:
                return "Perfect! This matches your target tolerance level"
            else:
                return "This dish is below your target level - good for building confidence"
                
        except Exception as e:
            logger.error(f"Error generating tolerance suggestion: {e}")
            return None
    
    def _convert_meituan_dish(self, meituan_dish: Dict[str, Any]) -> Optional[DishData]:
        """Convert Meituan dish format to internal format"""
        try:
            # Extract ingredients and classify as main vs spice
            all_ingredients = meituan_dish.get('ingredients', [])
            
            # Simple classification (would be more sophisticated in practice)
            spice_keywords = ['pepper', 'chili', 'spicy', 'hot', '辣', '椒', '麻']
            spice_ingredients = [ing for ing in all_ingredients 
                               if any(keyword in ing.lower() for keyword in spice_keywords)]
            main_ingredients = [ing for ing in all_ingredients if ing not in spice_ingredients]
            
            # Estimate spice level from name and ingredients
            estimated_scoville = self.data_collector.estimate_scoville_from_ingredients(
                spice_ingredients, meituan_dish.get('cooking_method', 'stir_fry')
            )
            
            dish = DishData(
                dish_id=meituan_dish['dish_id'],
                name=meituan_dish['name'],
                restaurant_id=meituan_dish['restaurant_id'],
                cuisine_type=meituan_dish.get('cuisine_type', 'Chinese'),
                main_ingredients=main_ingredients,
                cooking_method=meituan_dish.get('cooking_method', 'stir_fry'),
                spice_ingredients=spice_ingredients,
                lab_scoville=estimated_scoville,
                base_spice_level=self.data_collector.convert_scoville_to_scale(estimated_scoville),
                price=meituan_dish.get('price'),
                description=meituan_dish.get('description', '')
            )
            
            return dish
            
        except Exception as e:
            logger.error(f"Error converting Meituan dish: {e}")
            return None
    
    def _get_restaurant_dishes(self, restaurant_id: str) -> List[Dict[str, Any]]:
        """Get dishes for a restaurant"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT dish_id, name, cuisine_type, price, base_spice_level, description
        FROM dishes
        WHERE restaurant_id = ?
        ORDER BY name
        '''
        
        df = pd.read_sql_query(query, conn, params=(restaurant_id,))
        conn.close()
        
        return df.to_dict('records')
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Run the Flask API server"""
        self.app.run(host=host, port=port, debug=debug)

class MeituanWebhookHandler:
    """Handler for Meituan webhooks"""
    
    def __init__(self, api_instance: MeituanSpiceAPI):
        self.api = api_instance
        self.webhook_secret = "meituan_webhook_secret_2024"
    
    def verify_webhook_signature(self, payload: str, signature: str) -> bool:
        """Verify webhook signature for security"""
        expected_signature = hmac.new(
            self.webhook_secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    def handle_new_rating(self, data: Dict[str, Any]):
        """Handle new rating webhook from Meituan"""
        try:
            # Process new rating
            rating = UserRating(
                user_id=data['user_id'],
                dish_id=data['dish_id'],
                spice_rating=data['spice_rating'],
                overall_rating=data['overall_rating'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                review_text=data.get('review_text')
            )
            
            self.api.data_collector.add_user_rating(rating)
            self.api.user_profiler.update_profile_with_new_rating(
                rating.user_id, rating.dish_id, 
                rating.spice_rating, rating.overall_rating
            )
            
            logger.info(f"Processed new rating from user {rating.user_id}")
            
        except Exception as e:
            logger.error(f"Error handling new rating webhook: {e}")
    
    def handle_dish_update(self, data: Dict[str, Any]):
        """Handle dish update webhook from Meituan"""
        try:
            dish = self.api._convert_meituan_dish(data)
            if dish:
                self.api.data_collector.add_dish(dish)
                logger.info(f"Updated dish {dish.dish_id}")
                
        except Exception as e:
            logger.error(f"Error handling dish update webhook: {e}")

if __name__ == "__main__":
    # Example usage
    api = MeituanSpiceAPI()
    
    print("Starting Meituan Spice Prediction API...")
    print("Available endpoints:")
    print("  GET  /api/v1/health")
    print("  POST /api/v1/predict_spice")
    print("  GET/POST /api/v1/user/profile")
    print("  POST /api/v1/rating/submit")
    print("  GET/POST /api/v1/tolerance/plan")
    print("  POST /api/v1/dishes/sync")
    print("  GET  /api/v1/restaurants/<id>/dishes")
    print("  GET  /api/v1/analytics/bias")
    
    # Run API server
    api.run(host='0.0.0.0', port=5000, debug=True)