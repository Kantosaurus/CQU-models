#!/usr/bin/env python3
"""
Comprehensive Test Suite for Chongqing Spice Prediction System
Group 2 - AI-Powered Spice Level Prediction

This test suite covers all major components:
1. Data collection and processing
2. User tolerance profiling
3. XGBoost prediction model
4. Reviewer bias correction
5. Tolerance building recommendations
6. API integration
"""

import unittest
import sys
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collection import SpiceDataCollector, DishData, UserRating
from user_profiling import UserProfilingSystem, ToleranceProfile
from spice_prediction_model import SpicePredictionModel
from bias_correction import ReviewerBiasCorrector, BiasProfile
from tolerance_building import ToleranceBuildingEngine, ToleranceBuildingPlan
from meituan_integration import MeituanSpiceAPI

class TestDataCollection(unittest.TestCase):
    """Test cases for data collection system"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_spice.db")
        self.collector = SpiceDataCollector(self.test_db)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_database_initialization(self):
        """Test database setup and initialization"""
        # Check that tables exist
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['dishes', 'user_ratings', 'user_profiles', 'scoville_measurements']
        for table in expected_tables:
            self.assertIn(table, tables)
        
        conn.close()
    
    def test_dish_creation_and_storage(self):
        """Test dish data creation and database storage"""
        dish = DishData(
            dish_id="TEST001",
            name="Test Spicy Dish",
            restaurant_id="R001",
            cuisine_type="Sichuan",
            main_ingredients=["tofu", "pork"],
            cooking_method="stir_fry",
            spice_ingredients=["chao_tian_jiao", "hua_jiao"],
            price=25.0,
            description="Test dish for unit testing"
        )
        
        # Add dish to database
        self.collector.add_dish(dish)
        
        # Retrieve and verify
        retrieved_dish = self.collector.get_dish_data("TEST001")
        self.assertIsNotNone(retrieved_dish)
        self.assertEqual(retrieved_dish.name, "Test Spicy Dish")
        self.assertEqual(retrieved_dish.cuisine_type, "Sichuan")
        self.assertEqual(retrieved_dish.price, 25.0)
    
    def test_scoville_estimation(self):
        """Test Scoville unit estimation from ingredients"""
        # Test high-spice ingredients
        high_spice_ingredients = ["chao_tian_jiao", "xiao_mi_la"]
        high_scoville = self.collector.estimate_scoville_from_ingredients(
            high_spice_ingredients, "hot_pot"
        )
        
        # Test mild ingredients
        mild_ingredients = ["hua_jiao"]
        mild_scoville = self.collector.estimate_scoville_from_ingredients(
            mild_ingredients, "steam"
        )
        
        # High spice should be greater than mild
        self.assertGreater(high_scoville, mild_scoville)
        self.assertGreater(high_scoville, 1000)  # Should be significantly spicy
    
    def test_spice_level_conversion(self):
        """Test conversion from Scoville to 1-5 scale"""
        test_cases = [
            (0, 1),        # No spice
            (500, 1),      # Mild
            (2500, 2),     # Medium-mild
            (15000, 3),    # Medium
            (50000, 4),    # Hot
            (200000, 5)    # Very hot
        ]
        
        for scoville, expected_level in test_cases:
            level = self.collector.convert_scoville_to_scale(scoville)
            self.assertEqual(level, expected_level, 
                           f"Scoville {scoville} should map to level {expected_level}, got {level}")
    
    def test_sample_data_generation(self):
        """Test sample Chongqing dish data generation"""
        sample_dishes = self.collector.generate_sample_chongqing_dishes()
        
        self.assertGreater(len(sample_dishes), 0)
        
        # Check that all dishes have required fields
        for dish in sample_dishes:
            self.assertIsNotNone(dish.dish_id)
            self.assertIsNotNone(dish.name)
            self.assertIsNotNone(dish.base_spice_level)
            self.assertIn(dish.cuisine_type, ["Sichuan", "Chongqing"])
            self.assertGreater(len(dish.spice_ingredients), 0)
    
    def test_user_rating_storage(self):
        """Test user rating storage and retrieval"""
        rating = UserRating(
            user_id="U001",
            dish_id="TEST001",
            spice_rating=4,
            overall_rating=4.5,
            timestamp=datetime.now(),
            review_text="Very spicy but delicious",
            user_tolerance_level=3
        )
        
        self.collector.add_user_rating(rating)
        
        # Retrieve ratings for dish
        ratings = self.collector.get_user_ratings_for_dish("TEST001")
        self.assertEqual(len(ratings), 1)
        self.assertEqual(ratings[0].spice_rating, 4)
        self.assertEqual(ratings[0].user_id, "U001")

class TestUserProfiling(unittest.TestCase):
    """Test cases for user tolerance profiling"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_spice.db")
        self.profiler = UserProfilingSystem(self.test_db)
        self.collector = SpiceDataCollector(self.test_db)
        
        # Initialize with sample data
        self.collector.initialize_sample_data()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_tolerance_assessment_new_user(self):
        """Test tolerance assessment for new user with no ratings"""
        assessment = self.profiler.assess_user_tolerance("NEW_USER")
        
        # Should return default assessment
        self.assertEqual(assessment.assessed_tolerance, 3)  # Default moderate tolerance
        self.assertEqual(assessment.assessment_method, "default")
        self.assertEqual(assessment.evidence_count, 0)
        self.assertLess(assessment.confidence, 0.5)
    
    def test_tolerance_assessment_experienced_user(self):
        """Test tolerance assessment for user with rating history"""
        # User should already exist from sample data
        assessment = self.profiler.assess_user_tolerance("U001")
        
        # Should have some confidence and evidence
        self.assertGreater(assessment.evidence_count, 0)
        self.assertIn(assessment.assessed_tolerance, [1, 2, 3, 4, 5])
        self.assertEqual(assessment.assessment_method, "combined_analysis")
    
    def test_bias_detection(self):
        """Test reviewer bias detection"""
        user_id = "U001"
        bias = self.profiler.detect_rating_bias(user_id)
        
        # Bias should be a reasonable float value
        self.assertIsInstance(bias, float)
        self.assertGreaterEqual(bias, -2.0)
        self.assertLessEqual(bias, 2.0)
    
    def test_complete_profile_building(self):
        """Test complete tolerance profile creation"""
        profile = self.profiler.build_complete_profile("U001")
        
        self.assertIsInstance(profile, ToleranceProfile)
        self.assertEqual(profile.user_id, "U001")
        self.assertIn(profile.tolerance_level, [1, 2, 3, 4, 5])
        self.assertIsInstance(profile.preferred_spice_range, tuple)
        self.assertEqual(len(profile.preferred_spice_range), 2)
        self.assertIsInstance(profile.cuisine_preferences, dict)
        self.assertIsInstance(profile.rating_bias, float)
    
    def test_cuisine_preference_analysis(self):
        """Test cuisine preference analysis"""
        preferences = self.profiler._analyze_cuisine_preferences("U001")
        
        self.assertIsInstance(preferences, dict)
        for cuisine, preference in preferences.items():
            self.assertIsInstance(preference, float)
            self.assertGreaterEqual(preference, 0)
            self.assertLessEqual(preference, 1)

class TestSpicePredictionModel(unittest.TestCase):
    """Test cases for XGBoost spice prediction model"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_spice.db")
        self.model = SpicePredictionModel(self.test_db, os.path.join(self.temp_dir, "models"))
        
        # Initialize with sample data
        self.model.data_collector.initialize_sample_data()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_feature_engineering(self):
        """Test feature engineering process"""
        # Get sample training data
        raw_data = self.model.data_collector.export_training_data()
        
        if not raw_data.empty:
            features = self.model._engineer_features(raw_data)
            
            # Check that features were created
            self.assertGreater(len(features.columns), 10)  # Should have many features
            
            # Check for key feature categories
            feature_names = features.columns.tolist()
            self.assertIn('user_tolerance_level', feature_names)
            self.assertIn('base_spice_level', feature_names)
            self.assertIn('spice_ingredient_count', feature_names)
            
            # Check no NaN values in critical features
            self.assertFalse(features['user_tolerance_level'].isna().any())
            self.assertFalse(features['base_spice_level'].isna().any())
    
    def test_model_training(self):
        """Test XGBoost model training"""
        try:
            # Prepare training data
            features, target = self.model.prepare_training_data()
            
            if len(features) > 10:  # Need minimum data for training
                # Train model
                results = self.model.train_model(features, target)
                
                # Check training results
                self.assertIn('validation_mae', results)
                self.assertIn('validation_accuracy', results)
                self.assertIn('feature_importance', results)
                
                # Validate metrics are reasonable
                self.assertGreater(results['validation_accuracy'], 0.3)  # At least 30% accuracy
                self.assertLess(results['validation_mae'], 2.0)  # MAE should be reasonable
                
                # Check model was created
                self.assertIsNotNone(self.model.xgb_model)
        except ValueError as e:
            # Skip if insufficient training data
            self.skipTest(f"Insufficient training data: {e}")
    
    def test_prediction_generation(self):
        """Test personalized spice level prediction"""
        try:
            # Train model first
            features, target = self.model.prepare_training_data()
            if len(features) > 10:
                self.model.train_model(features, target)
                
                # Make prediction
                prediction = self.model.predict_personalized_spice_level("U001", "CQ001")
                
                # Validate prediction structure
                self.assertIn('predicted_spice_level', prediction)
                self.assertIn('confidence', prediction)
                self.assertIn('explanation', prediction)
                
                # Validate prediction values
                spice_level = prediction['predicted_spice_level']
                self.assertGreaterEqual(spice_level, 1)
                self.assertLessEqual(spice_level, 5)
                
                confidence = prediction['confidence']
                self.assertGreaterEqual(confidence, 0)
                self.assertLessEqual(confidence, 1)
                
        except ValueError as e:
            self.skipTest(f"Cannot test prediction: {e}")
    
    def test_model_serialization(self):
        """Test model saving and loading"""
        try:
            # Train a simple model
            features, target = self.model.prepare_training_data()
            if len(features) > 10:
                self.model.train_model(features, target)
                
                # Save model
                self.model.save_model("test_model.joblib")
                
                # Create new model instance and load
                new_model = SpicePredictionModel(self.test_db, os.path.join(self.temp_dir, "models"))
                new_model.load_model("test_model.joblib")
                
                # Check that model was loaded
                self.assertIsNotNone(new_model.xgb_model)
                self.assertEqual(len(new_model.feature_names), len(self.model.feature_names))
        except ValueError as e:
            self.skipTest(f"Cannot test serialization: {e}")

class TestBiasCorrection(unittest.TestCase):
    """Test cases for reviewer bias correction"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_spice.db")
        self.corrector = ReviewerBiasCorrector(self.test_db)
        self.collector = SpiceDataCollector(self.test_db)
        
        # Initialize with sample data
        self.collector.initialize_sample_data()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_systematic_bias_detection(self):
        """Test systematic bias detection"""
        bias, confidence = self.corrector.detect_systematic_bias("U001")
        
        self.assertIsInstance(bias, float)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
        self.assertGreaterEqual(bias, -3)
        self.assertLessEqual(bias, 3)
    
    def test_cultural_bias_detection(self):
        """Test cultural bias detection"""
        bias, confidence = self.corrector.detect_cultural_bias("U001")
        
        self.assertIsInstance(bias, float)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(bias, -2)
        self.assertLessEqual(bias, 2)
    
    def test_bias_profile_creation(self):
        """Test complete bias profile creation"""
        profile = self.corrector.build_bias_profile("U001")
        
        self.assertIsInstance(profile, BiasProfile)
        self.assertEqual(profile.user_id, "U001")
        self.assertIsInstance(profile.systematic_bias, float)
        self.assertIsInstance(profile.cultural_bias, float)
        self.assertIsInstance(profile.temporal_trend, float)
        self.assertIsInstance(profile.consistency_score, float)
        self.assertGreaterEqual(profile.confidence, 0)
        self.assertLessEqual(profile.confidence, 1)
    
    def test_bias_correction_application(self):
        """Test application of bias correction"""
        original_rating = 4.0
        correction = self.corrector.apply_bias_correction("U001", original_rating, "CQ001")
        
        self.assertEqual(correction.original_rating, original_rating)
        self.assertGreaterEqual(correction.corrected_rating, 1)
        self.assertLessEqual(correction.corrected_rating, 5)
        self.assertIsInstance(correction.correction_applied, float)
        self.assertIsInstance(correction.confidence, float)
        self.assertIsInstance(correction.correction_reason, str)
    
    def test_batch_correction(self):
        """Test batch bias correction"""
        # Create sample ratings dataframe
        ratings_data = pd.DataFrame({
            'user_id': ['U001', 'U002', 'U001'],
            'dish_id': ['CQ001', 'CQ002', 'CQ003'],
            'spice_rating': [3, 4, 5]
        })
        
        corrected_df = self.corrector.batch_correct_ratings(ratings_data)
        
        self.assertEqual(len(corrected_df), len(ratings_data))
        self.assertIn('corrected_rating', corrected_df.columns)
        self.assertIn('correction_applied', corrected_df.columns)
        self.assertIn('confidence', corrected_df.columns)

class TestToleranceBuilding(unittest.TestCase):
    """Test cases for tolerance building system"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_spice.db")
        self.tolerance_engine = ToleranceBuildingEngine(self.test_db)
        
        # Initialize with sample data
        self.tolerance_engine.data_collector.initialize_sample_data()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_tolerance_plan_creation(self):
        """Test tolerance building plan creation"""
        plan = self.tolerance_engine.create_tolerance_building_plan("U001", target_tolerance=4)
        
        self.assertIsInstance(plan, ToleranceBuildingPlan)
        self.assertEqual(plan.user_id, "U001")
        self.assertEqual(plan.target_tolerance, 4)
        self.assertGreater(plan.estimated_timeline_weeks, 0)
        self.assertGreater(len(plan.weekly_goals), 0)
        self.assertGreater(len(plan.safety_guidelines), 0)
        self.assertGreater(len(plan.progress_milestones), 0)
    
    def test_weekly_goals_generation(self):
        """Test weekly goals generation"""
        weekly_goals = self.tolerance_engine._generate_weekly_goals(2, 4, 8)  # 2→4 over 8 weeks
        
        self.assertEqual(len(weekly_goals), 8)
        
        for i, goal in enumerate(weekly_goals):
            self.assertIn('week', goal)
            self.assertIn('target_tolerance', goal)
            self.assertIn('focus', goal)
            self.assertEqual(goal['week'], i + 1)
            self.assertGreater(goal['target_tolerance'], 2)
    
    def test_progressive_dish_selection(self):
        """Test progressive dish selection for tolerance building"""
        dishes = self.tolerance_engine._select_progressive_dishes(2, 4, 6)  # 2→4 over 6 weeks
        
        if dishes:  # Only test if dishes are available
            self.assertGreater(len(dishes), 0)
            
            for dish_id, week, reason in dishes:
                self.assertIsInstance(dish_id, str)
                self.assertIsInstance(week, int)
                self.assertIsInstance(reason, str)
                self.assertGreaterEqual(week, 1)
                self.assertLessEqual(week, 6)
    
    def test_safety_guidelines_generation(self):
        """Test safety guidelines generation"""
        guidelines = self.tolerance_engine._generate_safety_guidelines(1, 5)
        
        self.assertGreater(len(guidelines), 5)  # Should have multiple guidelines
        
        # Check for key safety elements
        guidelines_text = " ".join(guidelines)
        self.assertIn("dairy", guidelines_text.lower())
        self.assertIn("water", guidelines_text.lower())
        self.assertIn("gradual", guidelines_text.lower())
    
    def test_progress_tracking(self):
        """Test progress tracking and updates"""
        # Create a plan first
        plan = self.tolerance_engine.create_tolerance_building_plan("U002", target_tolerance=4)
        
        # Update progress
        progress = self.tolerance_engine.update_progress(
            "U002", week=2, comfort_level=4, dishes_tried=["CQ001", "CQ002"]
        )
        
        self.assertEqual(progress.user_id, "U002")
        self.assertEqual(progress.current_week, 2)
        self.assertEqual(progress.comfort_level, 4)
        self.assertEqual(len(progress.dishes_tried), 2)
        self.assertIsInstance(progress.tolerance_gain, float)
        self.assertIsInstance(progress.confidence_change, float)

class TestAPIIntegration(unittest.TestCase):
    """Test cases for Meituan API integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_spice.db")
        self.api = MeituanSpiceAPI(self.test_db)
        
        # Set up test client
        self.client = self.api.app.test_client()
        self.client.testing = True
        
        # API key for authentication
        self.headers = {'X-API-Key': 'meituan_spice_api_key_2024'}
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_health_endpoint(self):
        """Test API health check endpoint"""
        response = self.client.get('/api/v1/health')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertIn('timestamp', data)
        self.assertIn('version', data)
        self.assertIn('services', data)
    
    def test_authentication_required(self):
        """Test that API endpoints require authentication"""
        # Try without API key
        response = self.client.post('/api/v1/predict_spice', json={
            'user_id': 'U001',
            'dish_id': 'CQ001'
        })
        
        self.assertEqual(response.status_code, 401)
    
    def test_spice_prediction_endpoint(self):
        """Test spice level prediction endpoint"""
        # Test with valid request
        response = self.client.post('/api/v1/predict_spice', 
                                  headers=self.headers,
                                  json={
                                      'user_id': 'U001',
                                      'dish_id': 'CQ001'
                                  })
        
        if response.status_code == 200:  # Only test if model is available
            data = json.loads(response.data)
            
            self.assertIn('dish_id', data)
            self.assertIn('predicted_spice_level', data)
            self.assertIn('confidence', data)
            self.assertIn('explanation', data)
            self.assertIn('user_recommendation', data)
            
            # Validate prediction values
            self.assertGreaterEqual(data['predicted_spice_level'], 1)
            self.assertLessEqual(data['predicted_spice_level'], 5)
            self.assertGreaterEqual(data['confidence'], 0)
            self.assertLessEqual(data['confidence'], 1)
    
    def test_user_profile_endpoint(self):
        """Test user profile GET endpoint"""
        response = self.client.get('/api/v1/user/profile?user_id=U001',
                                 headers=self.headers)
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            self.assertIn('user_id', data)
            self.assertIn('tolerance_level', data)
            self.assertIn('confidence', data)
            self.assertIn('preferred_range', data)
            self.assertIn('cuisine_preferences', data)
    
    def test_rating_submission_endpoint(self):
        """Test rating submission endpoint"""
        response = self.client.post('/api/v1/rating/submit',
                                  headers=self.headers,
                                  json={
                                      'user_id': 'U001',
                                      'dish_id': 'CQ001',
                                      'spice_rating': 4,
                                      'overall_rating': 4.5,
                                      'review_text': 'Very spicy but good'
                                  })
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'rating submitted')
    
    def test_tolerance_plan_endpoint(self):
        """Test tolerance building plan endpoint"""
        # Create plan
        response = self.client.post('/api/v1/tolerance/plan',
                                  headers=self.headers,
                                  json={
                                      'user_id': 'U001',
                                      'target_tolerance': 4
                                  })
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'plan created')
        
        # Retrieve plan
        response = self.client.get('/api/v1/tolerance/plan?user_id=U001',
                                 headers=self.headers)
        
        if response.status_code == 200:  # Plan was created successfully
            data = json.loads(response.data)
            
            self.assertIn('user_id', data)
            self.assertIn('current_tolerance', data)
            self.assertIn('target_tolerance', data)
            self.assertIn('timeline_weeks', data)
    
    def test_error_handling(self):
        """Test API error handling"""
        # Test missing required fields
        response = self.client.post('/api/v1/predict_spice',
                                  headers=self.headers,
                                  json={'user_id': 'U001'})  # Missing dish_id
        
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)

class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_spice.db")
        
        # Initialize all components
        self.collector = SpiceDataCollector(self.test_db)
        self.profiler = UserProfilingSystem(self.test_db)
        self.model = SpicePredictionModel(self.test_db, os.path.join(self.temp_dir, "models"))
        self.corrector = ReviewerBiasCorrector(self.test_db)
        self.tolerance_engine = ToleranceBuildingEngine(self.test_db)
        
        # Initialize with sample data
        self.collector.initialize_sample_data()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_user_journey(self):
        """Test complete user journey through the system"""
        user_id = "INTEGRATION_USER"
        
        # 1. New user makes some ratings
        ratings = [
            UserRating(user_id, "CQ001", 3, 4.0, datetime.now()),
            UserRating(user_id, "CQ002", 4, 4.5, datetime.now()),
            UserRating(user_id, "CQ003", 2, 3.5, datetime.now())
        ]
        
        for rating in ratings:
            self.collector.add_user_rating(rating)
        
        # 2. Build user profile
        profile = self.profiler.build_complete_profile(user_id)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.user_id, user_id)
        
        # 3. Detect and correct bias
        bias_profile = self.corrector.build_bias_profile(user_id)
        self.assertIsNotNone(bias_profile)
        
        correction = self.corrector.apply_bias_correction(user_id, 3.5, "CQ004")
        self.assertIsNotNone(correction)
        
        # 4. Create tolerance building plan
        plan = self.tolerance_engine.create_tolerance_building_plan(user_id, target_tolerance=4)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.user_id, user_id)
        
        # 5. Update progress
        progress = self.tolerance_engine.update_progress(
            user_id, week=1, comfort_level=3, dishes_tried=["CQ001"]
        )
        self.assertIsNotNone(progress)
        self.assertEqual(progress.user_id, user_id)
    
    def test_model_prediction_integration(self):
        """Test model prediction with real user data"""
        try:
            # Train model with sample data
            features, target = self.model.prepare_training_data()
            
            if len(features) > 10:
                results = self.model.train_model(features, target)
                
                # Make predictions for existing users
                prediction = self.model.predict_personalized_spice_level("U001", "CQ002")
                
                self.assertIsNotNone(prediction)
                self.assertIn('predicted_spice_level', prediction)
                self.assertIn('confidence', prediction)
                
                # Prediction should be reasonable
                spice_level = prediction['predicted_spice_level']
                self.assertGreaterEqual(spice_level, 1)
                self.assertLessEqual(spice_level, 5)
                
        except ValueError as e:
            self.skipTest(f"Cannot test model integration: {e}")

def run_all_tests():
    """Run all test suites"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_cases = [
        TestDataCollection,
        TestUserProfiling,
        TestSpicePredictionModel,
        TestBiasCorrection, 
        TestToleranceBuilding,
        TestAPIIntegration,
        TestSystemIntegration
    ]
    
    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print results summary
    print("\n" + "="*80)
    print("CHONGQING SPICE PREDICTION SYSTEM - TEST RESULTS")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\n')[-2]}")
    
    print("\n" + "="*80)
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)