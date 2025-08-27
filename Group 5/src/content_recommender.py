from typing import List, Dict, Tuple, Set
import numpy as np
from collections import defaultdict, Counter
from src.product_manager import Product, ProductManager
from src.customer_manager import Customer, CustomerManager
from datetime import datetime, timedelta

class ContentBasedRecommender:
    """
    Content-based recommendation system for recurring elderly customers.
    Analyzes purchase history and recommends healthier/better alternatives.
    """
    
    def __init__(self, product_manager: ProductManager, customer_manager: CustomerManager):
        self.product_manager = product_manager
        self.customer_manager = customer_manager

    def get_customer_profile(self, customer_id: str) -> Dict[str, float]:
        """Build a customer profile based on purchase history"""
        customer = self.customer_manager.get_customer(customer_id)
        if not customer or not customer.shopping_history:
            return {}
        
        profile = {
            'categories': defaultdict(float),
            'price_range': {'min': float('inf'), 'max': 0, 'avg': 0},
            'nutritional_preferences': defaultdict(float),
            'brand_preferences': defaultdict(float)
        }
        
        total_purchases = 0
        total_price = 0
        
        for record in customer.shopping_history:
            for product_id in record['products']:
                product = self.product_manager.get_product(product_id)
                if product:
                    # Category preferences
                    profile['categories'][product.category] += 1
                    
                    # Price analysis
                    profile['price_range']['min'] = min(profile['price_range']['min'], product.price)
                    profile['price_range']['max'] = max(profile['price_range']['max'], product.price)
                    total_price += product.price
                    total_purchases += 1
                    
                    # Nutritional preferences
                    for nutrient, value in product.nutritional_content.items():
                        profile['nutritional_preferences'][nutrient] += value
                    
                    # Brand preferences
                    if product.brand:
                        profile['brand_preferences'][product.brand] += 1
        
        # Normalize values
        if total_purchases > 0:
            profile['price_range']['avg'] = total_price / total_purchases
            
            # Normalize category preferences
            for category in profile['categories']:
                profile['categories'][category] /= total_purchases
            
            # Normalize nutritional preferences
            for nutrient in profile['nutritional_preferences']:
                profile['nutritional_preferences'][nutrient] /= total_purchases
        
        return profile

    def predict_next_purchases(self, customer_id: str, limit: int = 10) -> List[Tuple[Product, float]]:
        """Predict what the customer might want to buy based on purchase patterns"""
        customer = self.customer_manager.get_customer(customer_id)
        if not customer:
            return []
        
        # Get frequently bought items
        product_frequency = defaultdict(int)
        for record in customer.shopping_history:
            for product_id in record['products']:
                product_frequency[product_id] += 1
        
        # Sort by frequency and recency
        recent_purchases = []
        if customer.shopping_history:
            # Consider purchases from last 3 shopping trips
            recent_records = customer.shopping_history[-3:]
            recent_products = set()
            for record in recent_records:
                recent_products.update(record['products'])
        
        predictions = []
        customer_profile = self.get_customer_profile(customer_id)
        
        # Find products in preferred categories that haven't been bought recently
        for category, preference_score in customer_profile['categories'].items():
            category_products = self.product_manager.get_products_by_category(category)
            
            for product in category_products:
                # Skip if bought recently
                if product.product_id in recent_products:
                    continue
                
                # Calculate prediction score
                score = preference_score
                
                # Price compatibility
                avg_price = customer_profile['price_range']['avg']
                if avg_price > 0:
                    price_diff = abs(product.price - avg_price) / avg_price
                    price_score = max(0, 1 - price_diff)
                    score *= price_score
                
                # Nutritional compatibility
                nutrition_score = self._calculate_nutritional_compatibility(
                    product, customer_profile['nutritional_preferences'])
                score *= (1 + nutrition_score)
                
                predictions.append((product, score))
        
        # Sort by prediction score
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:limit]

    def recommend_healthier_alternatives(self, customer_id: str, limit: int = 8) -> List[Tuple[Product, Product, str]]:
        """Recommend healthier alternatives to frequently bought items"""
        customer = self.customer_manager.get_customer(customer_id)
        if not customer:
            return []
        
        # Get frequently bought items
        product_frequency = Counter()
        for record in customer.shopping_history:
            for product_id in record['products']:
                product_frequency[product_id] += 1
        
        recommendations = []
        
        # For each frequently bought item, find healthier alternatives
        for product_id, frequency in product_frequency.most_common(10):
            original_product = self.product_manager.get_product(product_id)
            if not original_product:
                continue
            
            # Find healthier alternatives
            alternatives = self._find_healthier_alternatives(original_product)
            
            for alternative, reason in alternatives:
                if len(recommendations) < limit:
                    recommendations.append((original_product, alternative, reason))
        
        return recommendations

    def _find_healthier_alternatives(self, product: Product) -> List[Tuple[Product, str]]:
        """Find healthier alternatives for a given product"""
        alternatives = []
        original_score = product.get_nutritional_score()
        
        # Look in the same category first
        category_products = self.product_manager.get_products_by_category(product.category)
        
        for candidate in category_products:
            if candidate.product_id == product.product_id:
                continue
            
            candidate_score = candidate.get_nutritional_score()
            
            # Must be healthier and reasonably priced
            if candidate_score > original_score and candidate.price <= product.price * 1.3:
                reason = self._get_health_improvement_reason(product, candidate)
                alternatives.append((candidate, reason))
        
        # Sort by nutritional improvement
        alternatives.sort(key=lambda x: x[0].get_nutritional_score(), reverse=True)
        return alternatives[:3]

    def _get_health_improvement_reason(self, original: Product, alternative: Product) -> str:
        """Generate a reason why the alternative is healthier"""
        reasons = []
        
        orig_nutrition = original.nutritional_content
        alt_nutrition = alternative.nutritional_content
        
        # Check key nutritional improvements
        if alt_nutrition.get('protein', 0) > orig_nutrition.get('protein', 0) * 1.2:
            reasons.append("higher protein")
        
        if alt_nutrition.get('fiber', 0) > orig_nutrition.get('fiber', 0) * 1.2:
            reasons.append("more fiber")
        
        if alt_nutrition.get('sugar', 100) < orig_nutrition.get('sugar', 100) * 0.8:
            reasons.append("less sugar")
        
        if alt_nutrition.get('sodium', 1000) < orig_nutrition.get('sodium', 1000) * 0.8:
            reasons.append("lower sodium")
        
        if alt_nutrition.get('calcium', 0) > orig_nutrition.get('calcium', 0) * 1.2:
            reasons.append("more calcium")
        
        if reasons:
            return f"Better choice: {', '.join(reasons)}"
        else:
            return "Generally healthier nutritional profile"

    def _calculate_nutritional_compatibility(self, product: Product, 
                                          nutritional_preferences: Dict[str, float]) -> float:
        """Calculate how well a product matches customer's nutritional preferences"""
        if not nutritional_preferences:
            return 0
        
        compatibility_score = 0
        total_nutrients = 0
        
        for nutrient, customer_avg in nutritional_preferences.items():
            if nutrient in product.nutritional_content:
                product_value = product.nutritional_content[nutrient]
                # Calculate similarity (closer to customer's average preference is better)
                if customer_avg > 0:
                    similarity = 1 - abs(product_value - customer_avg) / customer_avg
                    compatibility_score += max(0, similarity)
                    total_nutrients += 1
        
        return compatibility_score / max(total_nutrients, 1)

    def get_personalized_recommendations(self, customer_id: str, limit: int = 10) -> List[Tuple[Product, float, str]]:
        """Get personalized recommendations combining prediction and health optimization"""
        recommendations = []
        
        # Get predictions
        predictions = self.predict_next_purchases(customer_id, limit * 2)
        
        # Get healthier alternatives
        healthier_alternatives = self.recommend_healthier_alternatives(customer_id, limit)
        
        # Combine and prioritize
        seen_products = set()
        
        # Add predicted items
        for product, score in predictions:
            if product.product_id not in seen_products:
                recommendations.append((product, score, "Based on your shopping pattern"))
                seen_products.add(product.product_id)
        
        # Add healthier alternatives
        for original, alternative, reason in healthier_alternatives:
            if alternative.product_id not in seen_products:
                recommendations.append((alternative, 0.8, reason))
                seen_products.add(alternative.product_id)
        
        # Sort by score and limit results
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:limit]

    def analyze_shopping_patterns(self, customer_id: str) -> Dict[str, any]:
        """Analyze customer's shopping patterns for insights"""
        customer = self.customer_manager.get_customer(customer_id)
        if not customer:
            return {}
        
        analysis = {
            'total_shopping_trips': len(customer.shopping_history),
            'avg_items_per_trip': 0,
            'most_frequent_categories': [],
            'price_trends': [],
            'health_score_trend': [],
            'recommendations': []
        }
        
        if not customer.shopping_history:
            return analysis
        
        # Calculate average items per trip
        total_items = sum(len(record['products']) for record in customer.shopping_history)
        analysis['avg_items_per_trip'] = total_items / len(customer.shopping_history)
        
        # Most frequent categories
        category_counts = defaultdict(int)
        for record in customer.shopping_history:
            for product_id in record['products']:
                product = self.product_manager.get_product(product_id)
                if product:
                    category_counts[product.category] += 1
        
        analysis['most_frequent_categories'] = sorted(
            category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Health and price trends over time
        for record in customer.shopping_history:
            trip_price = 0
            trip_health_score = 0
            item_count = 0
            
            for product_id in record['products']:
                product = self.product_manager.get_product(product_id)
                if product:
                    trip_price += product.price
                    trip_health_score += product.get_nutritional_score()
                    item_count += 1
            
            analysis['price_trends'].append(trip_price)
            if item_count > 0:
                analysis['health_score_trend'].append(trip_health_score / item_count)
        
        return analysis