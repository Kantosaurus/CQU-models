#!/usr/bin/env python3
"""
Unit tests for Smart Cart Robot System
Group 5 - CQU AI Project
"""

import unittest
import sys
import os
import numpy as np
from datetime import datetime
import tempfile
import shutil

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from customer_manager import Customer, CustomerManager
from product_manager import Product, ProductManager
from popularity_recommender import PopularityRecommender
from content_recommender import ContentBasedRecommender
from recommendation_engine import SmartCartRecommendationEngine

class TestCustomerManager(unittest.TestCase):
    """Test cases for Customer Management System"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.customer_manager = CustomerManager(
            data_file=os.path.join(self.temp_dir, "test_customers.pkl"))
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_customer_creation(self):
        """Test customer object creation"""
        face_encoding = np.random.rand(128)
        customer = Customer("test_001", face_encoding, "Test User")
        
        self.assertEqual(customer.customer_id, "test_001")
        self.assertEqual(customer.name, "Test User")
        self.assertEqual(len(customer.shopping_history), 0)
        self.assertIsNotNone(customer.created_date)
    
    def test_customer_registration(self):
        """Test customer registration process"""
        # Create a dummy face image (RGB format)
        face_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Mock face recognition (would normally detect faces)
        # For testing, we'll directly add a customer
        face_encoding = np.random.rand(128)
        customer = Customer("test_customer", face_encoding, "Test Customer")
        self.customer_manager.customers["test_customer"] = customer
        
        # Test customer retrieval
        retrieved_customer = self.customer_manager.get_customer("test_customer")
        self.assertIsNotNone(retrieved_customer)
        self.assertEqual(retrieved_customer.customer_id, "test_customer")
    
    def test_purchase_history(self):
        """Test purchase history functionality"""
        face_encoding = np.random.rand(128)
        customer = Customer("test_002", face_encoding)
        self.customer_manager.customers["test_002"] = customer
        
        # Add purchase history
        products = ["P001", "P002", "P003"]
        self.customer_manager.add_purchase_history("test_002", products)
        
        updated_customer = self.customer_manager.get_customer("test_002")
        self.assertEqual(len(updated_customer.shopping_history), 1)
        self.assertEqual(updated_customer.shopping_history[0]['products'], products)
    
    def test_new_customer_detection(self):
        """Test new customer detection"""
        face_encoding = np.random.rand(128)
        customer = Customer("new_customer", face_encoding)
        self.customer_manager.customers["new_customer"] = customer
        
        # Should be detected as new customer (no purchase history)
        self.assertTrue(self.customer_manager.is_new_customer("new_customer"))
        
        # Add purchase history
        self.customer_manager.add_purchase_history("new_customer", ["P001"])
        
        # Should no longer be detected as new customer
        self.assertFalse(self.customer_manager.is_new_customer("new_customer"))

class TestProductManager(unittest.TestCase):
    """Test cases for Product Management System"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.product_manager = ProductManager(
            data_file=os.path.join(self.temp_dir, "test_products.pkl"))
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_product_creation(self):
        """Test product object creation"""
        nutritional_content = {"calories": 100, "protein": 5, "sugar": 10}
        product = Product("P999", "Test Product", "test_category", 5.99, nutritional_content)
        
        self.assertEqual(product.product_id, "P999")
        self.assertEqual(product.name, "Test Product")
        self.assertEqual(product.category, "test_category")
        self.assertEqual(product.price, 5.99)
        self.assertEqual(product.nutritional_content, nutritional_content)
    
    def test_nutritional_score(self):
        """Test nutritional score calculation"""
        # High protein, low sugar product
        good_nutrition = {"protein": 20, "fiber": 10, "sugar": 2, "sodium": 100}
        good_product = Product("P001", "Healthy Food", "test", 5.0, good_nutrition)
        
        # Low protein, high sugar product
        poor_nutrition = {"protein": 2, "fiber": 1, "sugar": 25, "sodium": 800}
        poor_product = Product("P002", "Junk Food", "test", 3.0, poor_nutrition)
        
        good_score = good_product.get_nutritional_score()
        poor_score = poor_product.get_nutritional_score()
        
        self.assertGreater(good_score, poor_score)
    
    def test_category_filtering(self):
        """Test product filtering by category"""
        # Add test products
        dairy_product = Product("D001", "Milk", "dairy", 4.0, {"protein": 8})
        fruit_product = Product("F001", "Apple", "fruits", 2.0, {"fiber": 4})
        
        self.product_manager.products["D001"] = dairy_product
        self.product_manager.products["F001"] = fruit_product
        self.product_manager._build_category_map()
        
        dairy_products = self.product_manager.get_products_by_category("dairy")
        self.assertEqual(len(dairy_products), 1)
        self.assertEqual(dairy_products[0].product_id, "D001")
    
    def test_popular_products(self):
        """Test popular products retrieval"""
        # Create products with different sales numbers
        product1 = Product("P001", "Popular Item", "test", 5.0, {"protein": 5}, monthly_sales=1000)
        product2 = Product("P002", "Less Popular", "test", 4.0, {"protein": 3}, monthly_sales=100)
        
        self.product_manager.products["P001"] = product1
        self.product_manager.products["P002"] = product2
        
        popular_products = self.product_manager.get_popular_products(limit=2)
        
        # Should be sorted by sales (descending)
        self.assertEqual(popular_products[0].product_id, "P001")
        self.assertEqual(popular_products[1].product_id, "P002")
    
    def test_healthier_alternatives(self):
        """Test finding healthier alternatives"""
        # Original product with poor nutrition
        original = Product("P001", "Regular Bread", "grains", 3.0, 
                         {"protein": 4, "sugar": 8, "fiber": 2})
        
        # Healthier alternative
        healthier = Product("P002", "Whole Grain Bread", "grains", 3.5,
                          {"protein": 8, "sugar": 3, "fiber": 6})
        
        self.product_manager.products["P001"] = original
        self.product_manager.products["P002"] = healthier
        self.product_manager._build_category_map()
        
        alternatives = self.product_manager.get_healthier_alternatives("P001")
        
        self.assertGreater(len(alternatives), 0)
        # Healthier product should have better nutritional score
        self.assertGreater(alternatives[0].get_nutritional_score(), 
                          original.get_nutritional_score())

class TestRecommendationSystems(unittest.TestCase):
    """Test cases for Recommendation Systems"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize managers
        self.customer_manager = CustomerManager(
            data_file=os.path.join(self.temp_dir, "test_customers.pkl"))
        self.product_manager = ProductManager(
            data_file=os.path.join(self.temp_dir, "test_products.pkl"))
        
        # Initialize recommenders
        self.popularity_recommender = PopularityRecommender(
            self.product_manager, self.customer_manager)
        self.content_recommender = ContentBasedRecommender(
            self.product_manager, self.customer_manager)
        
        # Add test customer with purchase history
        face_encoding = np.random.rand(128)
        customer = Customer("test_customer", face_encoding, "Test Customer")
        customer.add_purchase(["P001", "P004", "P007"])  # Milk, Bananas, Bread
        customer.add_purchase(["P002", "P005"])          # Yogurt, Spinach
        self.customer_manager.customers["test_customer"] = customer
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_popularity_recommendations(self):
        """Test popularity-based recommendations for new users"""
        recommendations = self.popularity_recommender.get_popular_recommendations(limit=5)
        
        self.assertGreater(len(recommendations), 0)
        self.assertLessEqual(len(recommendations), 5)
        
        # Check that recommendations are tuples of (Product, score)
        for product, score in recommendations:
            self.assertIsInstance(product, Product)
            self.assertIsInstance(score, (int, float))
            self.assertGreater(score, 0)
    
    def test_essential_items_for_elderly(self):
        """Test essential items recommendation for elderly"""
        essential_items = self.popularity_recommender.get_essential_items_for_elderly()
        
        self.assertGreater(len(essential_items), 0)
        
        # Check format: (Product, reason)
        for product, reason in essential_items:
            self.assertIsInstance(product, Product)
            self.assertIsInstance(reason, str)
            self.assertIn("health", reason.lower())
    
    def test_budget_friendly_recommendations(self):
        """Test budget-friendly recommendations"""
        budget_recommendations = self.popularity_recommender.get_budget_friendly_recommendations(
            max_price=8.0, limit=5)
        
        # All recommendations should be within budget
        for product, score in budget_recommendations:
            self.assertLessEqual(product.price, 8.0)
    
    def test_content_based_recommendations(self):
        """Test content-based recommendations for returning users"""
        personalized_recs = self.content_recommender.get_personalized_recommendations(
            "test_customer", limit=5)
        
        self.assertGreater(len(personalized_recs), 0)
        
        # Check format: (Product, score, reason)
        for product, score, reason in personalized_recs:
            self.assertIsInstance(product, Product)
            self.assertIsInstance(score, (int, float))
            self.assertIsInstance(reason, str)
    
    def test_customer_profile_building(self):
        """Test customer profile generation"""
        profile = self.content_recommender.get_customer_profile("test_customer")
        
        self.assertIn('categories', profile)
        self.assertIn('price_range', profile)
        self.assertGreater(len(profile['categories']), 0)
    
    def test_healthier_alternatives_recommendation(self):
        """Test healthier alternatives for frequent purchases"""
        healthier_alternatives = self.content_recommender.recommend_healthier_alternatives(
            "test_customer", limit=3)
        
        # Check format: (original_product, alternative_product, reason)
        for original, alternative, reason in healthier_alternatives:
            self.assertIsInstance(original, Product)
            self.assertIsInstance(alternative, Product)
            self.assertIsInstance(reason, str)
            # Alternative should be healthier
            self.assertGreater(alternative.get_nutritional_score(), 
                              original.get_nutritional_score())

class TestRecommendationEngine(unittest.TestCase):
    """Test cases for the main Recommendation Engine"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.engine = SmartCartRecommendationEngine(data_dir=self.temp_dir + "/")
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
        self.engine.end_session()
    
    def test_engine_initialization(self):
        """Test recommendation engine initialization"""
        self.assertIsNotNone(self.engine.customer_manager)
        self.assertIsNotNone(self.engine.product_manager)
        self.assertIsNotNone(self.engine.popularity_recommender)
        self.assertIsNotNone(self.engine.content_recommender)
    
    def test_customer_session_management(self):
        """Test customer session start and end"""
        # Start session without camera
        customer_id = self.engine.start_customer_session(use_camera=False)
        
        if customer_id:  # Only test if session started successfully
            self.assertIsNotNone(self.engine.current_customer)
            self.assertEqual(self.engine.current_customer, customer_id)
            
            # End session
            self.engine.end_session()
            self.assertIsNone(self.engine.current_customer)
    
    def test_recommendation_generation(self):
        """Test recommendation generation for different scenarios"""
        # Mock a customer session
        self.engine.current_customer = "test_customer"
        
        # Create test customer
        face_encoding = np.random.rand(128)
        customer = Customer("test_customer", face_encoding)
        self.engine.customer_manager.customers["test_customer"] = customer
        
        # Test recommendations for new customer
        new_recommendations = self.engine.get_recommendations(recommendation_type="popular")
        self.assertGreater(len(new_recommendations), 0)
        
        # Add purchase history to make customer "returning"
        customer.add_purchase(["P001", "P002"])
        
        # Test recommendations for returning customer
        returning_recommendations = self.engine.get_recommendations(recommendation_type="personalized")
        self.assertGreater(len(returning_recommendations), 0)
    
    def test_cart_functionality(self):
        """Test adding items to cart"""
        # Mock customer session
        self.engine.current_customer = "test_customer"
        face_encoding = np.random.rand(128)
        customer = Customer("test_customer", face_encoding)
        self.engine.customer_manager.customers["test_customer"] = customer
        
        # Test adding valid products
        valid_products = list(self.engine.product_manager.products.keys())[:3]
        result = self.engine.add_to_cart(valid_products)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["total_added"], len(valid_products))
    
    def test_shopping_summary(self):
        """Test shopping summary generation"""
        # Mock customer session with purchase
        self.engine.current_customer = "test_customer"
        face_encoding = np.random.rand(128)
        customer = Customer("test_customer", face_encoding)
        customer.add_purchase(["P001", "P002", "P003"])
        self.engine.customer_manager.customers["test_customer"] = customer
        self.engine.session_start_time = datetime.now()
        
        summary = self.engine.get_shopping_summary()
        
        self.assertIn("customer_id", summary)
        self.assertIn("session_duration", summary)
        self.assertIn("total_shopping_trips", summary)
        self.assertEqual(summary["customer_id"], "test_customer")
    
    def test_system_stats(self):
        """Test system statistics generation"""
        stats = self.engine.get_system_stats()
        
        self.assertIn("customer_stats", stats)
        self.assertIn("product_categories", stats)
        self.assertIn("total_products", stats)
        self.assertIn("active_session", stats)

class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.engine = SmartCartRecommendationEngine(data_dir=self.temp_dir + "/")
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
        self.engine.end_session()
    
    def test_complete_shopping_flow(self):
        """Test complete shopping flow from start to finish"""
        # 1. Start customer session
        self.engine.current_customer = "integration_test_customer"
        face_encoding = np.random.rand(128)
        customer = Customer("integration_test_customer", face_encoding, "Test Customer")
        self.engine.customer_manager.customers["integration_test_customer"] = customer
        self.engine.session_start_time = datetime.now()
        
        # 2. Get initial recommendations
        recommendations = self.engine.get_recommendations()
        self.assertGreater(len(recommendations), 0)
        
        # 3. Add items to cart
        selected_items = [rec[0].product_id for rec in recommendations[:3]]
        cart_result = self.engine.add_to_cart(selected_items)
        self.assertEqual(cart_result["status"], "success")
        
        # 4. Get updated recommendations after cart addition
        updated_recommendations = self.engine.get_recommendations()
        self.assertGreater(len(updated_recommendations), 0)
        
        # 5. Get shopping summary
        summary = self.engine.get_shopping_summary()
        self.assertEqual(summary["customer_id"], "integration_test_customer")
        self.assertGreater(summary["current_session_items"], 0)
        
        # 6. End session
        self.engine.end_session()
        self.assertIsNone(self.engine.current_customer)
    
    def test_new_vs_returning_customer_experience(self):
        """Test different experiences for new vs returning customers"""
        # Test new customer
        new_customer_id = "new_customer_test"
        face_encoding = np.random.rand(128)
        new_customer = Customer(new_customer_id, face_encoding)
        self.engine.customer_manager.customers[new_customer_id] = new_customer
        self.engine.current_customer = new_customer_id
        
        new_customer_recs = self.engine.get_recommendations()
        
        # Test returning customer
        returning_customer_id = "returning_customer_test"
        returning_customer = Customer(returning_customer_id, face_encoding)
        returning_customer.add_purchase(["P001", "P002", "P003"])
        returning_customer.add_purchase(["P004", "P005"])
        self.engine.customer_manager.customers[returning_customer_id] = returning_customer
        self.engine.current_customer = returning_customer_id
        
        returning_customer_recs = self.engine.get_recommendations()
        
        # Both should have recommendations, but they may differ
        self.assertGreater(len(new_customer_recs), 0)
        self.assertGreater(len(returning_customer_recs), 0)

def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_cases = [
        TestCustomerManager,
        TestProductManager, 
        TestRecommendationSystems,
        TestRecommendationEngine,
        TestIntegrationScenarios
    ]
    
    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print results summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2]}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)