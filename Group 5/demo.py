#!/usr/bin/env python3
"""
Smart Cart Robot - Demo Script
Group 5 - AI-Powered Grocery Shopping Assistant for Elderly

This demo script showcases the smart cart robot's AI capabilities:
1. Face recognition for customer identification
2. Popularity-based recommendations for new customers
3. Content-based recommendations for returning customers
4. Voice command processing
5. Healthier alternative suggestions
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from recommendation_engine import SmartCartRecommendationEngine

def print_separator():
    print("=" * 80)

def print_recommendations(recommendations, title="Recommendations"):
    """Print recommendations in a formatted way"""
    print(f"\n{title}:")
    print("-" * 40)
    
    if not recommendations:
        print("No recommendations available.")
        return
    
    for i, (product, score, reason) in enumerate(recommendations, 1):
        print(f"{i}. {product.name}")
        print(f"   Category: {product.category} | Price: ${product.price:.2f}")
        print(f"   Reason: {reason}")
        if hasattr(product, 'nutritional_content') and product.nutritional_content:
            nutrition_str = ", ".join([f"{k}: {v}" for k, v in list(product.nutritional_content.items())[:3]])
            print(f"   Nutrition: {nutrition_str}")
        print(f"   Health Score: {product.get_nutritional_score():.1f}")
        print()

def demo_new_customer_flow():
    """Demonstrate the flow for a new elderly customer"""
    print_separator()
    print("DEMO: New Elderly Customer Experience")
    print_separator()
    
    # Initialize the recommendation engine
    engine = SmartCartRecommendationEngine()
    
    print("🛒 Smart Cart Robot: Welcome to our smart grocery cart!")
    print("📷 Activating face recognition...")
    
    # Simulate new customer registration
    print("\n👤 New customer detected!")
    customer_id = engine.start_customer_session(use_camera=False)
    
    if not customer_id:
        print("❌ Customer session could not be started")
        return
    
    print(f"✅ Welcome, Customer {customer_id}!")
    print("🎯 Since you're new, I'll show you popular items among elderly customers in Chongqing.")
    
    # Get popularity-based recommendations
    recommendations = engine.get_recommendations(recommendation_type="popular")
    print_recommendations(recommendations, "Popular Items Among Elderly Customers")
    
    # Show essential healthy items
    essential_recommendations = engine.get_recommendations(recommendation_type="healthier")
    print_recommendations(essential_recommendations[:5], "Essential Health Items for Elderly")
    
    # Simulate adding items to cart
    print("\n🛍️ Customer selects some items...")
    selected_items = [rec[0].product_id for rec in recommendations[:3]]
    cart_result = engine.add_to_cart(selected_items)
    
    print(f"✅ {cart_result['message']}")
    print(f"   Added: {', '.join(cart_result['added_products'])}")
    
    # Get shopping summary
    summary = engine.get_shopping_summary()
    print(f"\n📊 Shopping Summary:")
    print(f"   Items in cart: {summary.get('current_session_items', 0)}")
    print(f"   Session total: ${summary.get('session_total_price', 0):.2f}")
    print(f"   Average health score: {summary.get('avg_health_score', 0):.1f}")
    
    engine.end_session()
    print("\n👋 Thank you for shopping with Smart Cart Robot!")

def demo_returning_customer_flow():
    """Demonstrate the flow for a returning elderly customer"""
    print_separator()
    print("DEMO: Returning Elderly Customer Experience")
    print_separator()
    
    engine = SmartCartRecommendationEngine()
    
    print("📷 Face recognition activated...")
    print("👤 Customer recognized: Mrs. Chen (frequent shopper)")
    
    # Simulate returning customer with purchase history
    customer_id = "customer_0001"
    
    # Create a customer with purchase history
    dummy_encoding = np.random.rand(128)
    from customer_manager import Customer
    customer = Customer(customer_id, dummy_encoding, "Mrs. Chen")
    
    # Add some purchase history
    customer.add_purchase(["P001", "P004", "P007", "P010"])  # Milk, Bananas, Bread, Chicken
    customer.add_purchase(["P002", "P005", "P008", "P011"])  # Yogurt, Spinach, Rice, Salmon
    customer.add_purchase(["P001", "P006", "P009", "P013"])  # Milk, Carrots, Oats, Green Tea
    
    engine.customer_manager.customers[customer_id] = customer
    engine.current_customer = customer_id
    engine.session_start_time = datetime.now()
    
    print("✅ Welcome back, Mrs. Chen!")
    print("🎯 Based on your shopping history, here are personalized recommendations:")
    
    # Get personalized recommendations
    personalized_recs = engine.get_recommendations(recommendation_type="personalized")
    print_recommendations(personalized_recs, "Personalized Recommendations")
    
    # Show healthier alternatives
    print("\n💚 I also found some healthier alternatives to items you usually buy:")
    healthier_recs = engine.get_recommendations(recommendation_type="healthier")
    print_recommendations(healthier_recs, "Healthier Alternatives")
    
    # Demonstrate voice command processing
    print("\n🎤 Voice command demo:")
    print("Customer: 'Find some dairy products'")
    
    # Simulate voice command
    import sys
    from io import StringIO
    old_stdin = sys.stdin
    sys.stdin = StringIO("find dairy products\n")
    
    try:
        voice_result = engine.process_voice_command()
        print(f"🤖 Smart Cart Response: {voice_result['message']}")
        if voice_result['recommendations']:
            print_recommendations(voice_result['recommendations'][:3], "Voice Search Results")
    finally:
        sys.stdin = old_stdin
    
    # Show customer analytics
    from content_recommender import ContentBasedRecommender
    content_rec = ContentBasedRecommender(engine.product_manager, engine.customer_manager)
    analysis = content_rec.analyze_shopping_patterns(customer_id)
    
    print("\n📈 Your Shopping Analytics:")
    print(f"   Total shopping trips: {analysis.get('total_shopping_trips', 0)}")
    print(f"   Average items per trip: {analysis.get('avg_items_per_trip', 0):.1f}")
    print(f"   Favorite categories: {', '.join([cat for cat, count in analysis.get('most_frequent_categories', [])[:3]])}")
    
    engine.end_session()
    print("\n👋 Thank you for shopping with us, Mrs. Chen! Have a healthy day!")

def demo_system_features():
    """Demonstrate additional system features"""
    print_separator()
    print("DEMO: System Features Overview")
    print_separator()
    
    engine = SmartCartRecommendationEngine()
    
    # Show system statistics
    stats = engine.get_system_stats()
    print("📊 System Statistics:")
    print(f"   Total customers: {stats['customer_stats']['total_customers']}")
    print(f"   New customers: {stats['customer_stats']['new_customers']}")
    print(f"   Returning customers: {stats['customer_stats']['returning_customers']}")
    print(f"   Total products: {stats['total_products']}")
    print(f"   Product categories: {', '.join(stats['product_categories'])}")
    
    # Show product categories and their statistics
    category_stats = engine.product_manager.get_category_stats()
    print("\n📦 Product Categories:")
    for category, stat in category_stats.items():
        print(f"   {category.title()}: {stat['count']} products, avg price ${stat['avg_price']:.2f}")
    
    # Demonstrate budget-friendly recommendations
    print("\n💰 Budget-Friendly Recommendations (under $8):")
    budget_recs = engine.popularity_recommender.get_budget_friendly_recommendations(max_price=8.0)
    print_recommendations(budget_recs[:5], "Budget-Friendly Options")
    
    # Show essential items for elderly
    print("\n🏥 Essential Items for Elderly Health:")
    essential_items = engine.popularity_recommender.get_essential_items_for_elderly()
    for product, reason in essential_items[:5]:
        print(f"   • {product.name} - {reason}")
        print(f"     Price: ${product.price:.2f}, Health Score: {product.get_nutritional_score():.1f}")
        print()
    
    engine.end_session()

def interactive_demo():
    """Interactive demo where user can choose what to explore"""
    print_separator()
    print("🤖 SMART CART ROBOT - Interactive Demo")
    print("AI-Powered Grocery Shopping Assistant for Elderly")
    print("Group 5 - CQU Project")
    print_separator()
    
    while True:
        print("\nChoose a demo to run:")
        print("1. 👶 New Customer Experience")
        print("2. 🔄 Returning Customer Experience")
        print("3. ⚙️ System Features Overview")
        print("4. 🏃 Run All Demos")
        print("5. ❌ Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            demo_new_customer_flow()
        elif choice == "2":
            demo_returning_customer_flow()
        elif choice == "3":
            demo_system_features()
        elif choice == "4":
            demo_new_customer_flow()
            demo_returning_customer_flow()
            demo_system_features()
        elif choice == "5":
            print("👋 Thank you for exploring Smart Cart Robot!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
        
        if choice in ["1", "2", "3", "4"]:
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        interactive_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your Python environment and dependencies.")