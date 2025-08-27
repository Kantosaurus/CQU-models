from typing import List, Dict, Tuple
from collections import Counter
from src.product_manager import Product, ProductManager
from src.customer_manager import CustomerManager

class PopularityRecommender:
    """
    Popularity-based collaborative filtering for new elderly customers.
    Recommends top-selling items among elderly customers in Chongqing.
    """
    
    def __init__(self, product_manager: ProductManager, customer_manager: CustomerManager):
        self.product_manager = product_manager
        self.customer_manager = customer_manager
        self.elderly_preferences = self._analyze_elderly_preferences()

    def _analyze_elderly_preferences(self) -> Dict[str, float]:
        """Analyze preferences specific to elderly customers"""
        # Categories that elderly customers typically prefer
        elderly_friendly_categories = {
            'dairy': 1.2,      # Good for bone health
            'fruits': 1.3,     # Vitamins and fiber
            'vegetables': 1.3, # Nutrients and fiber
            'fish': 1.4,       # Omega-3 for brain health
            'grains': 1.1,     # Energy and fiber
            'beverages': 0.9,  # Lower priority
            'protein': 1.2,    # Important for muscle health
            'meat': 1.0,       # Moderate priority
            'snacks': 0.7      # Lower priority for health
        }
        return elderly_friendly_categories

    def get_popular_recommendations(self, limit: int = 10, 
                                  exclude_categories: List[str] = None) -> List[Tuple[Product, float]]:
        """
        Get popular product recommendations for new elderly customers.
        
        Args:
            limit: Maximum number of recommendations
            exclude_categories: Categories to exclude from recommendations
            
        Returns:
            List of tuples (Product, relevance_score)
        """
        exclude_categories = exclude_categories or []
        recommendations = []
        
        # Get all products and calculate popularity scores
        for product in self.product_manager.products.values():
            if product.category in exclude_categories:
                continue
                
            # Calculate base popularity score from monthly sales
            popularity_score = product.monthly_sales
            
            # Apply elderly preference multiplier
            category_multiplier = self.elderly_preferences.get(product.category, 1.0)
            popularity_score *= category_multiplier
            
            # Apply health bonus (nutritional score bonus)
            health_bonus = product.get_nutritional_score() * 0.1
            popularity_score += health_bonus
            
            # Price consideration (elderly on fixed income prefer reasonable prices)
            if product.price < 5.0:
                price_bonus = 1.2
            elif product.price < 10.0:
                price_bonus = 1.0
            else:
                price_bonus = 0.8
            
            popularity_score *= price_bonus
            
            recommendations.append((product, popularity_score))
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:limit]

    def get_category_recommendations(self, category: str, limit: int = 5) -> List[Tuple[Product, float]]:
        """Get popular recommendations within a specific category"""
        category_products = self.product_manager.get_products_by_category(category)
        recommendations = []
        
        for product in category_products:
            # Score based on popularity and health
            score = product.monthly_sales + (product.get_nutritional_score() * 10)
            recommendations.append((product, score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:limit]

    def get_essential_items_for_elderly(self) -> List[Tuple[Product, str]]:
        """Get essential items specifically recommended for elderly health"""
        essential_items = []
        
        # High-calcium items for bone health
        dairy_products = self.product_manager.get_products_by_category('dairy')
        for product in dairy_products:
            if product.nutritional_content.get('calcium', 0) > 200:
                essential_items.append((product, "High in calcium for bone health"))
        
        # High-fiber items for digestive health
        for product in self.product_manager.products.values():
            if product.nutritional_content.get('fiber', 0) > 5:
                essential_items.append((product, "High fiber for digestive health"))
        
        # Omega-3 rich items for brain health
        fish_products = self.product_manager.get_products_by_category('fish')
        for product in fish_products:
            if product.nutritional_content.get('omega3', 0) > 1.0:
                essential_items.append((product, "Rich in Omega-3 for brain health"))
        
        # Low-sodium options for heart health
        for product in self.product_manager.products.values():
            if product.nutritional_content.get('sodium', 1000) < 200:
                essential_items.append((product, "Low sodium for heart health"))
        
        return essential_items[:10]

    def get_budget_friendly_recommendations(self, max_price: float = 8.0, 
                                          limit: int = 8) -> List[Tuple[Product, float]]:
        """Get popular recommendations within a budget (good for fixed-income elderly)"""
        affordable_products = self.product_manager.get_products_by_price_range(0, max_price)
        recommendations = []
        
        for product in affordable_products:
            # Score based on value (nutrition per dollar)
            value_score = product.get_nutritional_score() / max(product.price, 0.1)
            popularity_factor = product.monthly_sales / 100
            
            total_score = value_score + popularity_factor
            recommendations.append((product, total_score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:limit]

    def get_seasonal_recommendations(self, season: str = "current") -> List[Tuple[Product, str]]:
        """Get seasonal recommendations (placeholder for seasonal logic)"""
        # This is a simplified version - in practice, you'd have seasonal data
        seasonal_items = []
        
        if season.lower() in ["winter", "current"]:
            # Winter recommendations for elderly
            warm_beverages = [p for p in self.product_manager.products.values() 
                            if 'tea' in p.name.lower()]
            for product in warm_beverages:
                seasonal_items.append((product, "Warming beverage for winter"))
            
            # Vitamin C rich fruits
            citrus_products = [p for p in self.product_manager.products.values() 
                             if any(fruit in p.name.lower() for fruit in ['orange', 'lemon'])]
            for product in citrus_products:
                seasonal_items.append((product, "Vitamin C boost for immunity"))
        
        return seasonal_items

    def generate_shopping_list_suggestions(self, budget: float = 50.0) -> Dict[str, List[Product]]:
        """Generate a complete shopping list suggestion within budget"""
        shopping_list = {
            'essentials': [],
            'proteins': [],
            'fruits_vegetables': [],
            'dairy': [],
            'grains': []
        }
        
        remaining_budget = budget
        
        # Essential items first (30% of budget)
        essential_budget = budget * 0.3
        essentials = self.get_essential_items_for_elderly()
        
        for product, reason in essentials:
            if product.price <= remaining_budget and product.price <= essential_budget:
                shopping_list['essentials'].append(product)
                remaining_budget -= product.price
                essential_budget -= product.price
                if essential_budget <= 0:
                    break
        
        # Fill remaining categories with popular items
        categories = ['meat', 'fish', 'protein']  # Proteins
        for category in categories:
            popular_in_category = self.get_category_recommendations(category, 2)
            for product, score in popular_in_category:
                if product.price <= remaining_budget:
                    shopping_list['proteins'].append(product)
                    remaining_budget -= product.price
                    break
        
        # Add fruits and vegetables
        for category in ['fruits', 'vegetables']:
            popular_in_category = self.get_category_recommendations(category, 2)
            for product, score in popular_in_category:
                if product.price <= remaining_budget:
                    shopping_list['fruits_vegetables'].append(product)
                    remaining_budget -= product.price
        
        return shopping_list