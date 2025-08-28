#!/usr/bin/env python3
"""
Chongqing Spice Prediction System - Interactive Demo
Group 2 - AI-Powered Personalized Spice Level Prediction

This demo showcases the complete spice prediction system:
1. Personalized spice level predictions using XGBoost
2. User tolerance profiling and bias correction
3. Tolerance building recommendations
4. Real-time spice level rating for Chongqing dishes
"""

import sys
import os
import time
from datetime import datetime, timedelta
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection import SpiceDataCollector
from user_profiling import UserProfilingSystem
from spice_prediction_model import SpicePredictionModel
from bias_correction import ReviewerBiasCorrector
from tolerance_building import ToleranceBuildingEngine
from meituan_integration import MeituanSpiceAPI

def print_header(title: str):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def print_subheader(title: str):
    """Print formatted subsection header"""
    print("\n" + "-"*50)
    print(f" {title}")
    print("-"*50)

def demo_data_initialization():
    """Demonstrate data collection and initialization"""
    print_header("🌶️ CHONGQING SPICE PREDICTION SYSTEM DEMO")
    print("Welcome to the AI-powered spice level prediction system!")
    print("This system helps spice-minded diners understand spice levels")
    print("and build their tolerance progressively.")
    
    print_subheader("Data Initialization")
    print("📊 Initializing sample Chongqing food data...")
    
    # Initialize data collector
    collector = SpiceDataCollector()
    collector.initialize_sample_data()
    
    print("✅ Sample data initialized successfully!")
    
    # Show sample dishes
    training_data = collector.export_training_data()
    unique_dishes = training_data['dish_name'].unique()
    
    print(f"📋 Database contains {len(unique_dishes)} Chongqing dishes:")
    for i, dish in enumerate(unique_dishes[:8], 1):
        dish_data = training_data[training_data['dish_name'] == dish].iloc[0]
        spice_level = dish_data['base_spice_level']
        price = dish_data['price']
        print(f"   {i}. {dish} - Spice Level: {spice_level}/5, Price: ¥{price:.0f}")
    
    if len(unique_dishes) > 8:
        print(f"   ... and {len(unique_dishes) - 8} more dishes")
    
    print(f"\n📈 Database contains {len(training_data)} user ratings from {training_data['user_id'].nunique()} users")
    
    return collector

def demo_user_profiling(collector):
    """Demonstrate user tolerance profiling"""
    print_header("👤 USER TOLERANCE PROFILING")
    
    profiler = UserProfilingSystem(collector.db_path)
    
    print("🔍 Analyzing user tolerance profiles...")
    
    # Analyze a few sample users
    sample_users = ["U001", "U002", "U003"]
    
    for user_id in sample_users:
        print_subheader(f"User Profile: {user_id}")
        
        # Build complete profile
        profile = profiler.build_complete_profile(user_id)
        
        print(f"🌡️  Tolerance Level: {profile.tolerance_level}/5")
        print(f"📊 Confidence Score: {profile.confidence_score:.2f}")
        print(f"🎯 Preferred Spice Range: {profile.preferred_spice_range[0]}-{profile.preferred_spice_range[1]}")
        print(f"⚖️  Rating Bias: {profile.rating_bias:+.2f}")
        print(f"📈 Tolerance Trend: {profile.tolerance_trend:+.3f}")
        
        print("🍜 Cuisine Preferences:")
        for cuisine, preference in profile.cuisine_preferences.items():
            print(f"   • {cuisine}: {preference:.2f}")
        
        time.sleep(1)  # Pause for readability
    
    return profiler

def demo_bias_correction(collector):
    """Demonstrate reviewer bias detection and correction"""
    print_header("⚖️ REVIEWER BIAS CORRECTION")
    
    corrector = ReviewerBiasCorrector(collector.db_path)
    
    print("🔍 Detecting and correcting reviewer biases...")
    
    # Analyze bias patterns
    print_subheader("System-Wide Bias Analysis")
    
    bias_analysis = corrector.analyze_bias_patterns()
    
    if 'error' not in bias_analysis:
        print(f"👥 Total Users Analyzed: {bias_analysis['total_users_analyzed']}")
        print(f"📊 Average Systematic Bias: {bias_analysis['avg_systematic_bias']:+.3f}")
        print(f"🌍 Average Cultural Bias: {bias_analysis['avg_cultural_bias']:+.3f}")
        print(f"⏱️  Average Temporal Trend: {bias_analysis['avg_temporal_trend']:+.3f}")
        print(f"🎯 Average Consistency: {bias_analysis['avg_consistency_score']:.3f}")
        print(f"⚠️  High Bias Users: {bias_analysis['high_bias_users']}")
        
        # Show bias distribution
        print("\n🔍 Systematic Bias Distribution:")
        dist = bias_analysis['systematic_bias_distribution']
        print(f"   Very Negative: {dist['very_negative']} users")
        print(f"   Negative: {dist['negative']} users")
        print(f"   Neutral: {dist['neutral']} users")
        print(f"   Positive: {dist['positive']} users")
        print(f"   Very Positive: {dist['very_positive']} users")
    
    # Demonstrate individual bias correction
    print_subheader("Individual Bias Correction Examples")
    
    test_ratings = [
        ("U001", 3.5, "CQ001"),
        ("U002", 4.0, "CQ002"),
        ("U003", 2.5, "CQ003")
    ]
    
    for user_id, original_rating, dish_id in test_ratings:
        print(f"\n👤 User {user_id} rates dish {dish_id}: {original_rating}")
        
        correction = corrector.apply_bias_correction(user_id, original_rating, dish_id)
        
        print(f"   Original Rating: {correction.original_rating}")
        print(f"   Corrected Rating: {correction.corrected_rating:.2f}")
        print(f"   Correction Applied: {correction.correction_applied:+.3f}")
        print(f"   Confidence: {correction.confidence:.2f}")
        print(f"   Reason: {correction.correction_reason}")
    
    return corrector

def demo_spice_prediction(collector, profiler):
    """Demonstrate personalized spice level prediction"""
    print_header("🤖 AI SPICE LEVEL PREDICTION")
    
    print("🔧 Training XGBoost model for personalized predictions...")
    
    model = SpicePredictionModel(collector.db_path)
    
    try:
        # Prepare and train model
        features, target = model.prepare_training_data()
        print(f"📊 Training data: {len(features)} samples, {len(features.columns)} features")
        
        results = model.train_model(features, target)
        
        print("✅ Model training completed!")
        print(f"   📉 Validation MAE: {results['validation_mae']:.3f}")
        print(f"   🎯 Validation Accuracy: {results['validation_accuracy']:.3f}")
        print(f"   📊 Best Parameters: {results['best_params']}")
        
        # Show feature importance
        print("\n🔍 Top 10 Most Important Features:")
        top_features = results['feature_importance'].head(10)
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"   {i}. {row['feature']}: {row['importance']:.4f}")
        
        # Demonstrate predictions
        print_subheader("Personalized Spice Predictions")
        
        sample_predictions = [
            ("U001", "CQ001", "麻婆豆腐 (Mapo Tofu)"),
            ("U001", "CQ002", "重庆火锅 (Chongqing Hot Pot)"),
            ("U002", "CQ003", "口水鸡 (Saliva Chicken)"),
            ("U003", "CQ004", "辣子鸡 (Spicy Chicken)")
        ]
        
        for user_id, dish_id, dish_name in sample_predictions:
            print(f"\n🍜 {dish_name} for User {user_id}:")
            
            prediction = model.predict_personalized_spice_level(user_id, dish_id)
            
            print(f"   🌶️  Predicted Spice Level: {prediction['predicted_spice_level']:.1f}/5")
            print(f"   📊 Confidence: {prediction['confidence']:.2f}")
            print(f"   🎯 User Tolerance: {prediction['user_tolerance_level']}/5")
            print(f"   📏 Dish Base Spice: {prediction['dish_base_spice']}/5")
            print(f"   ⚖️  Bias Correction: {prediction['bias_correction']:+.3f}")
            print(f"   💡 Explanation: {prediction['explanation']}")
        
        return model
        
    except ValueError as e:
        print(f"❌ Could not train model: {e}")
        print("📝 This is normal for demo - insufficient training data")
        return None

def demo_tolerance_building(collector, profiler):
    """Demonstrate tolerance building recommendations"""
    print_header("📈 TOLERANCE BUILDING SYSTEM")
    
    tolerance_engine = ToleranceBuildingEngine(collector.db_path)
    
    print("🎯 Creating personalized tolerance building plans...")
    
    # Create tolerance building plans for different scenarios
    scenarios = [
        ("U001", 4, "Moderate to High Tolerance"),
        ("U002", 5, "High to Expert Level"),
        ("U003", 3, "Low to Moderate Tolerance")
    ]
    
    for user_id, target_tolerance, scenario_desc in scenarios:
        print_subheader(f"Scenario: {scenario_desc}")
        print(f"👤 User: {user_id}, Target: {target_tolerance}/5")
        
        try:
            plan = tolerance_engine.create_tolerance_building_plan(user_id, target_tolerance)
            
            print(f"📅 Timeline: {plan.estimated_timeline_weeks} weeks")
            print(f"🎯 Current → Target: {plan.current_tolerance} → {plan.target_tolerance}")
            print(f"🍽️  Recommended Dishes: {len(plan.recommended_dishes)} dishes")
            print(f"⚠️  Safety Guidelines: {len(plan.safety_guidelines)} guidelines")
            print(f"🏆 Milestones: {len(plan.progress_milestones)} checkpoints")
            
            # Show first few weekly goals
            print("\n📋 First 3 Weekly Goals:")
            for goal in plan.weekly_goals[:3]:
                print(f"   Week {goal['week']}: {goal['goal_description']}")
                print(f"     Target: {goal['target_tolerance']:.1f}/5")
                print(f"     Focus: {goal['focus']}")
                print(f"     Dishes to try: {goal['dishes_to_try']}")
            
            # Show safety highlights
            print("\n⚠️  Key Safety Guidelines:")
            for guideline in plan.safety_guidelines[:3]:
                print(f"   • {guideline}")
            
            # Simulate progress update
            print(f"\n📊 Simulating progress after 2 weeks...")
            progress = tolerance_engine.update_progress(
                user_id, week=2, comfort_level=4, 
                dishes_tried=["CQ001", "CQ003"], side_effects=[]
            )
            
            print(f"   📈 Tolerance Gain: {progress.tolerance_gain:+.2f}")
            print(f"   😊 Comfort Level: {progress.comfort_level}/5")
            print(f"   💪 Confidence Change: {progress.confidence_change:+.2f}")
            print(f"   🍜 Dishes Tried: {len(progress.dishes_tried)}")
            
        except Exception as e:
            print(f"❌ Could not create plan: {e}")
        
        time.sleep(1)
    
    return tolerance_engine

def demo_api_integration():
    """Demonstrate Meituan API integration"""
    print_header("🔌 MEITUAN API INTEGRATION")
    
    print("🚀 Starting Meituan Spice Prediction API...")
    
    try:
        api = MeituanSpiceAPI()
        
        print("✅ API initialized successfully!")
        print("\n📡 Available API Endpoints:")
        endpoints = [
            ("GET", "/api/v1/health", "System health check"),
            ("POST", "/api/v1/predict_spice", "Get personalized spice prediction"),
            ("GET/POST", "/api/v1/user/profile", "User tolerance profile management"),
            ("POST", "/api/v1/rating/submit", "Submit new spice rating"),
            ("GET/POST", "/api/v1/tolerance/plan", "Tolerance building plans"),
            ("POST", "/api/v1/dishes/sync", "Sync dish data from Meituan"),
            ("GET", "/api/v1/restaurants/<id>/dishes", "Get restaurant dishes"),
            ("GET", "/api/v1/analytics/bias", "Bias correction analytics")
        ]
        
        for method, endpoint, description in endpoints:
            print(f"   {method:10} {endpoint:35} - {description}")
        
        print("\n🔧 API Configuration:")
        print(f"   🔑 Authentication: API Key required")
        print(f"   ⏱️  Rate Limit: {api.rate_limit} requests/minute")
        print(f"   💾 Cache Duration: {api.cache_duration} seconds")
        print(f"   🤖 Model Status: {'✅ Loaded' if api.prediction_model.xgb_model else '⚠️ Not trained'}")
        
        # Demonstrate sample API usage
        print_subheader("Sample API Requests")
        
        sample_requests = [
            {
                "endpoint": "/api/v1/predict_spice",
                "method": "POST",
                "data": {"user_id": "U001", "dish_id": "CQ001"},
                "description": "Predict spice level for Mapo Tofu"
            },
            {
                "endpoint": "/api/v1/user/profile",
                "method": "GET", 
                "params": "user_id=U001",
                "description": "Get user tolerance profile"
            },
            {
                "endpoint": "/api/v1/tolerance/plan",
                "method": "POST",
                "data": {"user_id": "U001", "target_tolerance": 4},
                "description": "Create tolerance building plan"
            }
        ]
        
        for req in sample_requests:
            print(f"\n📝 {req['description']}:")
            print(f"   {req['method']} {req['endpoint']}")
            if 'data' in req:
                print(f"   Data: {req['data']}")
            if 'params' in req:
                print(f"   Params: {req['params']}")
            
            # Note: In a real demo, you would make actual HTTP requests here
            print("   💡 (Run API server separately to test actual requests)")
        
        print(f"\n🏃 To start the API server, run:")
        print(f"   python src/meituan_integration.py")
        print(f"   Server will be available at: http://localhost:5000")
        
        return api
        
    except Exception as e:
        print(f"❌ API initialization failed: {e}")
        return None

def demo_real_world_scenario():
    """Demonstrate a complete real-world usage scenario"""
    print_header("🌟 REAL-WORLD USAGE SCENARIO")
    
    print("👩 Scenario: Sarah, a tourist from Shanghai, wants to try Chongqing food")
    print("   but is worried about spice levels being too intense for her.")
    print("\n📱 She opens Meituan and sees spice predictions powered by our AI...")
    
    # Initialize components
    collector = SpiceDataCollector()
    if not os.path.exists(collector.db_path):
        collector.initialize_sample_data()
    
    profiler = UserProfilingSystem(collector.db_path)
    model = SpicePredictionModel(collector.db_path)
    
    # Simulate Sarah as a new user with low tolerance
    sarah_user_id = "SARAH_TOURIST"
    
    print_subheader("Step 1: Initial Assessment")
    print("🆕 Sarah is a new user - the system uses default moderate tolerance")
    print("📊 System creates initial profile based on regional background (Shanghai)")
    
    # Build initial profile
    try:
        profile = profiler.build_complete_profile(sarah_user_id)
        print(f"   Initial tolerance estimate: {profile.tolerance_level}/5")
        print(f"   Confidence: {profile.confidence_score:.2f}")
    except:
        print("   Using default profile for new user")
    
    print_subheader("Step 2: Menu Browsing with AI Predictions")
    print("🍜 Sarah browses a Chongqing restaurant menu...")
    
    # Simulate menu items with predictions
    menu_items = [
        ("CQ005", "酸辣粉 (Hot and Sour Noodles)", 2),
        ("CQ001", "麻婆豆腐 (Mapo Tofu)", 3),
        ("CQ002", "重庆火锅 (Chongqing Hot Pot)", 5),
        ("CQ004", "辣子鸡 (Spicy Chicken)", 4)
    ]
    
    try:
        # Try to make predictions if model is available
        features, target = model.prepare_training_data()
        if len(features) > 10:
            model.train_model(features, target)
            model_available = True
        else:
            model_available = False
    except:
        model_available = False
    
    for dish_id, dish_name, base_spice in menu_items:
        print(f"\n🍽️  {dish_name}")
        print(f"   Base Spice Level: {base_spice}/5")
        
        if model_available:
            try:
                prediction = model.predict_personalized_spice_level(sarah_user_id, dish_id)
                predicted_level = prediction['predicted_spice_level']
                confidence = prediction['confidence']
                explanation = prediction['explanation']
                
                print(f"   🤖 AI Prediction for Sarah: {predicted_level:.1f}/5")
                print(f"   📊 Confidence: {confidence:.2f}")
                print(f"   💡 {explanation}")
                
                # Generate recommendation
                if predicted_level <= 2.5:
                    rec = "🟢 Perfect for you! Should be comfortable."
                elif predicted_level <= 3.5:
                    rec = "🟡 Moderate spice - good to try with caution."
                elif predicted_level <= 4.5:
                    rec = "🟠 Quite spicy - consider sharing or having milk ready."
                else:
                    rec = "🔴 Very spicy - probably too intense for first visit."
                
                print(f"   💭 Recommendation: {rec}")
                
            except:
                # Fallback to simple recommendation
                if base_spice <= 2:
                    rec = "🟢 Should be mild for most people"
                elif base_spice <= 3:
                    rec = "🟡 Moderate spice level"
                else:
                    rec = "🟠 Quite spicy - caution advised"
                print(f"   💭 Recommendation: {rec}")
        else:
            # Simple rule-based recommendation
            if base_spice <= 2:
                rec = "🟢 Should be mild for most people"
            elif base_spice <= 3:
                rec = "🟡 Moderate spice level"
            else:
                rec = "🟠 Quite spicy - caution advised"
            print(f"   💭 Recommendation: {rec}")
    
    print_subheader("Step 3: Sarah Makes Her Choice")
    print("🎯 Sarah chooses '酸辣粉 (Hot and Sour Noodles)' - predicted as mild")
    print("🍜 She enjoys the meal and rates it 3/5 for spice")
    
    # Simulate rating submission
    from data_collection import UserRating
    rating = UserRating(
        user_id=sarah_user_id,
        dish_id="CQ005",
        spice_rating=3,
        overall_rating=4.2,
        timestamp=datetime.now(),
        review_text="Perfect introduction to Chongqing flavors!",
        user_tolerance_level=2  # Sarah realizes she's more sensitive
    )
    
    collector.add_user_rating(rating)
    print("✅ Rating submitted and profile updated")
    
    print_subheader("Step 4: Tolerance Building Suggestion")
    print("📈 The system notices Sarah enjoyed the experience and suggests tolerance building")
    
    tolerance_engine = ToleranceBuildingEngine(collector.db_path)
    
    try:
        plan = tolerance_engine.create_tolerance_building_plan(sarah_user_id, target_tolerance=3)
        print(f"🎯 Suggested plan: Build to tolerance level 3 over {plan.estimated_timeline_weeks} weeks")
        print("📅 Week 1 goal: Try similar mild dishes to build confidence")
        print("📅 Week 2 goal: Try dishes with slightly more spice")
        print("🏆 Final goal: Comfortably enjoy medium-spice Sichuan food")
        
        print("\n💡 First week recommendations:")
        for dish_id, week, reason in plan.recommended_dishes[:3]:
            if week <= 2:
                dish_data = collector.get_dish_data(dish_id)
                if dish_data:
                    print(f"   • {dish_data.name} - {reason}")
        
    except:
        print("🎯 Suggested plan: Gradually try spicier dishes over coming weeks")
        print("💡 Next recommendation: Try 麻婆豆腐 (Mapo Tofu) with extra rice")
    
    print_subheader("Impact Summary")
    print("✅ Sarah had a great first experience with Chongqing food")
    print("📊 Her rating helps improve predictions for other tourists")
    print("🎯 She has a clear path to build her spice tolerance")
    print("🏪 The restaurant gets better reviews from satisfied customers")
    print("🤝 Win-win for diners, restaurants, and the platform!")

def interactive_demo():
    """Main interactive demo"""
    print("🌶️  Welcome to the Chongqing Spice Prediction System Demo! 🌶️")
    print("\nThis comprehensive demo will walk you through all system components.")
    print("The demo takes about 5-10 minutes to complete.")
    
    input("\nPress Enter to start the demo...")
    
    # Component demos
    collector = demo_data_initialization()
    input("\nPress Enter to continue to user profiling...")
    
    profiler = demo_user_profiling(collector)
    input("\nPress Enter to continue to bias correction...")
    
    corrector = demo_bias_correction(collector)
    input("\nPress Enter to continue to AI prediction model...")
    
    model = demo_spice_prediction(collector, profiler)
    input("\nPress Enter to continue to tolerance building...")
    
    tolerance_engine = demo_tolerance_building(collector, profiler)
    input("\nPress Enter to continue to API integration...")
    
    api = demo_api_integration()
    input("\nPress Enter to see a real-world usage scenario...")
    
    demo_real_world_scenario()
    
    print_header("🎉 DEMO COMPLETE")
    print("Thank you for exploring the Chongqing Spice Prediction System!")
    print("\n📋 What you've seen:")
    print("   ✅ Comprehensive Chongqing food database with spice analysis")
    print("   ✅ Advanced user tolerance profiling with bias correction")
    print("   ✅ XGBoost AI model for personalized spice predictions")
    print("   ✅ Progressive tolerance building recommendations")
    print("   ✅ Full API integration ready for Meituan platform")
    print("   ✅ Real-world usage scenario demonstration")
    
    print("\n🚀 Next Steps:")
    print("   • Run the API server: python src/meituan_integration.py")
    print("   • Explore the code in the src/ directory")
    print("   • Run tests: python tests/test_spice_system.py")
    print("   • Read the comprehensive documentation in README.md")
    
    print("\n🌶️  Making Chongqing food accessible to everyone! 🌶️")

def quick_demo():
    """Quick 2-minute demo"""
    print("🌶️  QUICK DEMO: Chongqing Spice Prediction System")
    
    print("\n1️⃣  Initializing sample data...")
    collector = SpiceDataCollector()
    collector.initialize_sample_data()
    print("   ✅ 8 Chongqing dishes loaded with spice analysis")
    
    print("\n2️⃣  Building user tolerance profile...")
    profiler = UserProfilingSystem(collector.db_path)
    profile = profiler.build_complete_profile("U001")
    print(f"   ✅ User tolerance: {profile.tolerance_level}/5, Bias: {profile.rating_bias:+.2f}")
    
    print("\n3️⃣  Training AI prediction model...")
    model = SpicePredictionModel(collector.db_path)
    try:
        features, target = model.prepare_training_data()
        results = model.train_model(features, target)
        print(f"   ✅ Model trained! Accuracy: {results['validation_accuracy']:.2f}")
        
        # Make sample prediction
        prediction = model.predict_personalized_spice_level("U001", "CQ002")
        print(f"   🤖 Chongqing Hot Pot prediction: {prediction['predicted_spice_level']:.1f}/5")
        
    except:
        print("   ⚠️  Model training skipped (insufficient data)")
    
    print("\n4️⃣  Creating tolerance building plan...")
    tolerance_engine = ToleranceBuildingEngine(collector.db_path)
    plan = tolerance_engine.create_tolerance_building_plan("U001", 4)
    print(f"   ✅ Plan created: {plan.estimated_timeline_weeks} weeks to reach level 4")
    
    print("\n5️⃣  API ready for integration...")
    api = MeituanSpiceAPI()
    print("   ✅ 8 endpoints ready for Meituan integration")
    
    print(f"\n🎉 System ready! Perfect for helping diners navigate Chongqing's spicy cuisine.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_demo()
    else:
        interactive_demo()