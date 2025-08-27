from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
from src.customer_manager import CustomerManager
from src.product_manager import ProductManager, Product
from src.popularity_recommender import PopularityRecommender
from src.content_recommender import ContentBasedRecommender
from src.face_recognition_service import FaceRecognitionService, VoiceRecognitionService

class SmartCartRecommendationEngine:
    """
    Main recommendation engine for the smart cart robot.
    Coordinates all recommendation systems and provides unified interface.
    """
    
    def __init__(self, data_dir: str = "data/"):
        # Initialize core components
        self.customer_manager = CustomerManager(f"{data_dir}customers.pkl")
        self.product_manager = ProductManager(f"{data_dir}products.pkl")
        
        # Initialize recommendation systems
        self.popularity_recommender = PopularityRecommender(
            self.product_manager, self.customer_manager)
        self.content_recommender = ContentBasedRecommender(
            self.product_manager, self.customer_manager)
        
        # Initialize recognition services
        self.face_recognition = FaceRecognitionService(self.customer_manager)
        self.voice_recognition = VoiceRecognitionService()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Current session state
        self.current_customer = None
        self.session_start_time = None
        self.current_recommendations = []
        
    def start_customer_session(self, use_camera: bool = True) -> Optional[str]:
        """
        Start a new customer session with face recognition.
        
        Args:
            use_camera: Whether to use camera for face recognition
            
        Returns:
            Customer ID if successful, None otherwise
        """
        self.logger.info("Starting customer session...")
        
        if use_camera:
            # Try face recognition first
            customer_id = self.face_recognition.interactive_customer_recognition(
                timeout_seconds=30)
            
            if not customer_id:
                # Offer to register new customer
                print("Customer not recognized. Would you like to register? (y/n)")
                response = input().lower()
                if response == 'y':
                    name = input("Enter your name (optional): ")
                    frame = self.face_recognition.capture_frame()
                    if frame is not None:
                        customer_id = self.face_recognition.register_new_customer(
                            frame, name)
        else:
            # Manual customer ID input for testing
            customer_id = input("Enter customer ID for testing: ")
            if customer_id and not self.customer_manager.get_customer(customer_id):
                print("Customer not found. Creating new customer...")
                # Create a dummy customer for testing
                import numpy as np
                dummy_encoding = np.random.rand(128)
                self.customer_manager.customers[customer_id] = \
                    self.customer_manager.Customer(customer_id, dummy_encoding)
        
        if customer_id:
            self.current_customer = customer_id
            self.session_start_time = datetime.now()
            self.logger.info(f"Session started for customer: {customer_id}")
            
            # Generate initial recommendations
            self._generate_initial_recommendations()
        
        return customer_id
    
    def _generate_initial_recommendations(self):
        """Generate initial recommendations for the current customer"""
        if not self.current_customer:
            return
        
        is_new_customer = self.customer_manager.is_new_customer(self.current_customer)
        
        if is_new_customer:
            self.logger.info("Generating recommendations for new customer")
            recommendations = self.popularity_recommender.get_popular_recommendations(limit=10)
            self.current_recommendations = [
                (product, score, "Popular among elderly shoppers") 
                for product, score in recommendations
            ]
        else:
            self.logger.info("Generating personalized recommendations for returning customer")
            self.current_recommendations = \
                self.content_recommender.get_personalized_recommendations(
                    self.current_customer, limit=10)
    
    def get_recommendations(self, category: str = None, 
                          recommendation_type: str = "mixed") -> List[Tuple[Product, float, str]]:
        """
        Get recommendations for the current customer.
        
        Args:
            category: Filter by product category
            recommendation_type: "popular", "personalized", "healthier", or "mixed"
            
        Returns:
            List of (Product, score, reason) tuples
        """
        if not self.current_customer:
            return []
        
        recommendations = []
        is_new_customer = self.customer_manager.is_new_customer(self.current_customer)
        
        if recommendation_type == "popular" or (is_new_customer and recommendation_type == "mixed"):
            # Popularity-based recommendations
            popular_recs = self.popularity_recommender.get_popular_recommendations(limit=8)
            recommendations.extend([
                (product, score, "Popular among elderly shoppers")
                for product, score in popular_recs
            ])
            
        elif recommendation_type == "personalized" or (not is_new_customer and recommendation_type == "mixed"):
            # Content-based recommendations
            personalized_recs = self.content_recommender.get_personalized_recommendations(
                self.current_customer, limit=8)
            recommendations.extend(personalized_recs)
            
        elif recommendation_type == "healthier":
            # Healthier alternatives
            if not is_new_customer:
                healthier_recs = self.content_recommender.recommend_healthier_alternatives(
                    self.current_customer, limit=8)
                recommendations.extend([
                    (alternative, 0.9, reason)
                    for original, alternative, reason in healthier_recs
                ])
            else:
                # For new customers, show essential healthy items
                essential_items = self.popularity_recommender.get_essential_items_for_elderly()
                recommendations.extend([
                    (product, 0.9, reason)
                    for product, reason in essential_items[:8]
                ])
        
        # Filter by category if specified
        if category:
            recommendations = [
                (product, score, reason) for product, score, reason in recommendations
                if product.category.lower() == category.lower()
            ]
        
        # Sort by score and remove duplicates
        seen_products = set()
        unique_recommendations = []
        for product, score, reason in sorted(recommendations, key=lambda x: x[1], reverse=True):
            if product.product_id not in seen_products:
                unique_recommendations.append((product, score, reason))
                seen_products.add(product.product_id)
        
        return unique_recommendations[:10]
    
    def process_voice_command(self) -> Dict[str, any]:
        """
        Process voice commands and return appropriate response.
        
        Returns:
            Dictionary with command results and recommendations
        """
        command = self.voice_recognition.process_voice_command()
        if not command:
            return {"status": "error", "message": "No command received"}
        
        command_lower = command.lower()
        result = {"status": "success", "command": command, "recommendations": []}
        
        # Search for specific products
        if "find" in command_lower or "search" in command_lower:
            # Extract search term
            search_terms = command_lower.replace("find", "").replace("search", "").strip()
            search_results = self.product_manager.search_products(search_terms)
            result["recommendations"] = [
                (product, 1.0, "Search result") for product in search_results[:5]
            ]
            result["message"] = f"Found {len(search_results)} products matching '{search_terms}'"
        
        # Show recommendations
        elif "recommend" in command_lower or "suggest" in command_lower:
            result["recommendations"] = self.get_recommendations()
            result["message"] = "Here are my recommendations for you"
        
        # Show healthier alternatives
        elif "health" in command_lower or "better" in command_lower:
            result["recommendations"] = self.get_recommendations(recommendation_type="healthier")
            result["message"] = "Here are some healthier options"
        
        # Category-specific requests
        elif any(category in command_lower for category in ["fruit", "vegetable", "dairy", "meat", "grain"]):
            for category in ["fruits", "vegetables", "dairy", "meat", "grains"]:
                if category[:-1] in command_lower:  # Handle singular forms
                    result["recommendations"] = self.get_recommendations(category=category)
                    result["message"] = f"Here are {category} recommendations"
                    break
        
        else:
            result["recommendations"] = self.get_recommendations()
            result["message"] = "Here are some recommendations based on your request"
        
        return result
    
    def add_to_cart(self, product_ids: List[str]) -> Dict[str, any]:
        """
        Add products to customer's cart and update purchase history.
        
        Args:
            product_ids: List of product IDs to add to cart
            
        Returns:
            Dictionary with operation results
        """
        if not self.current_customer:
            return {"status": "error", "message": "No active customer session"}
        
        valid_products = []
        invalid_products = []
        
        for product_id in product_ids:
            product = self.product_manager.get_product(product_id)
            if product:
                valid_products.append(product)
                # Update product sales
                self.product_manager.update_sales(product_id)
            else:
                invalid_products.append(product_id)
        
        if valid_products:
            # Update customer purchase history
            valid_product_ids = [p.product_id for p in valid_products]
            self.customer_manager.add_purchase_history(
                self.current_customer, valid_product_ids)
            
            # Generate new recommendations based on cart
            self._generate_cart_based_recommendations(valid_products)
        
        return {
            "status": "success",
            "added_products": [p.name for p in valid_products],
            "invalid_products": invalid_products,
            "total_added": len(valid_products),
            "message": f"Added {len(valid_products)} items to cart"
        }
    
    def _generate_cart_based_recommendations(self, cart_products: List[Product]):
        """Generate recommendations based on current cart contents"""
        cart_categories = set(p.category for p in cart_products)
        
        # Suggest complementary items
        complementary_recommendations = []
        
        for category in cart_categories:
            # Get similar products in the same category
            for product in cart_products:
                if product.category == category:
                    similar_products = self.product_manager.get_similar_products(
                        product.product_id, limit=2)
                    for similar in similar_products:
                        complementary_recommendations.append(
                            (similar, 0.8, f"Goes well with {product.name}"))
        
        # Add to current recommendations
        self.current_recommendations.extend(complementary_recommendations[:5])
    
    def get_shopping_summary(self) -> Dict[str, any]:
        """
        Get summary of current shopping session.
        
        Returns:
            Dictionary with shopping session summary
        """
        if not self.current_customer:
            return {"status": "error", "message": "No active session"}
        
        customer = self.customer_manager.get_customer(self.current_customer)
        if not customer:
            return {"status": "error", "message": "Customer not found"}
        
        # Get latest purchase (current session)
        latest_purchase = customer.shopping_history[-1] if customer.shopping_history else None
        
        summary = {
            "customer_id": self.current_customer,
            "session_duration": str(datetime.now() - self.session_start_time) if self.session_start_time else "N/A",
            "is_new_customer": len(customer.shopping_history) <= 1,
            "total_shopping_trips": len(customer.shopping_history),
            "current_session_items": 0,
            "recommendations_shown": len(self.current_recommendations),
            "customer_preferences": customer.get_category_preferences() if customer.shopping_history else {}
        }
        
        if latest_purchase:
            summary["current_session_items"] = len(latest_purchase["products"])
            
            # Calculate session total
            session_total = 0
            session_health_score = 0
            for product_id in latest_purchase["products"]:
                product = self.product_manager.get_product(product_id)
                if product:
                    session_total += product.price
                    session_health_score += product.get_nutritional_score()
            
            summary["session_total_price"] = session_total
            summary["avg_health_score"] = session_health_score / len(latest_purchase["products"]) if latest_purchase["products"] else 0
        
        return summary
    
    def end_session(self):
        """End the current customer session"""
        if self.current_customer:
            self.logger.info(f"Ending session for customer: {self.current_customer}")
            
            # Save any pending data
            self.customer_manager.save_customers()
            self.product_manager.save_products()
            
            # Clean up
            self.current_customer = None
            self.session_start_time = None
            self.current_recommendations = []
            
            # Clean up camera resources
            self.face_recognition.cleanup()
    
    def get_system_stats(self) -> Dict[str, any]:
        """Get system statistics and health metrics"""
        return {
            "customer_stats": self.customer_manager.get_customer_stats(),
            "product_categories": list(self.product_manager.category_map.keys()),
            "total_products": len(self.product_manager.products),
            "active_session": self.current_customer is not None,
            "current_customer": self.current_customer
        }