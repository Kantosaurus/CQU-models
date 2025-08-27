import pickle
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class Product:
    product_id: str
    name: str
    category: str
    price: float
    nutritional_content: Dict[str, float]  # {'calories': 100, 'sugar': 5, 'protein': 2, 'fat': 1}
    monthly_sales: int = 0
    description: str = ""
    brand: str = ""

    def get_nutritional_score(self) -> float:
        """Calculate a simple nutritional score (higher is healthier)"""
        protein = self.nutritional_content.get('protein', 0)
        fiber = self.nutritional_content.get('fiber', 0)
        sugar = self.nutritional_content.get('sugar', 0)
        sodium = self.nutritional_content.get('sodium', 0)
        
        # Higher protein and fiber is good, lower sugar and sodium is good
        score = (protein * 2) + (fiber * 1.5) - (sugar * 0.5) - (sodium * 0.001)
        return max(0, score)  # Ensure non-negative score

class ProductManager:
    def __init__(self, data_file: str = "data/products.pkl"):
        self.data_file = data_file
        self.products: Dict[str, Product] = {}
        self.category_map: Dict[str, List[str]] = {}
        self.load_products()
        self._build_category_map()

    def load_products(self):
        try:
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                self.products = data.get('products', {})
        except (FileNotFoundError, EOFError):
            print("No existing product data found. Initializing with sample data.")
            self._initialize_sample_data()

    def save_products(self):
        import os
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        with open(self.data_file, 'wb') as f:
            pickle.dump({'products': self.products}, f)

    def _initialize_sample_data(self):
        """Initialize with sample grocery products relevant to elderly customers"""
        sample_products = [
            # Dairy
            Product("P001", "Low-Fat Milk 1L", "dairy", 4.50, 
                   {"calories": 150, "protein": 8, "sugar": 12, "fat": 2.5, "calcium": 300}, 450),
            Product("P002", "Greek Yogurt", "dairy", 3.20, 
                   {"calories": 100, "protein": 15, "sugar": 6, "fat": 0, "calcium": 200}, 320),
            Product("P003", "Low-Sodium Cheese", "dairy", 5.80, 
                   {"calories": 110, "protein": 7, "sugar": 1, "fat": 9, "sodium": 140}, 180),
            
            # Fruits & Vegetables
            Product("P004", "Bananas (1kg)", "fruits", 3.00, 
                   {"calories": 89, "protein": 1, "sugar": 12, "fiber": 3, "potassium": 358}, 600),
            Product("P005", "Spinach (300g)", "vegetables", 2.50, 
                   {"calories": 23, "protein": 3, "sugar": 0, "fiber": 2, "iron": 2.7}, 200),
            Product("P006", "Carrots (1kg)", "vegetables", 2.80, 
                   {"calories": 41, "protein": 1, "sugar": 5, "fiber": 3, "vitamin_a": 835}, 380),
            
            # Grains & Bread
            Product("P007", "Whole Wheat Bread", "grains", 2.90, 
                   {"calories": 247, "protein": 13, "sugar": 4, "fiber": 7, "sodium": 400}, 420),
            Product("P008", "Brown Rice (2kg)", "grains", 6.50, 
                   {"calories": 216, "protein": 5, "sugar": 0, "fiber": 4, "magnesium": 84}, 290),
            Product("P009", "Oats (1kg)", "grains", 4.20, 
                   {"calories": 389, "protein": 17, "sugar": 1, "fiber": 11, "beta_glucan": 4}, 250),
            
            # Protein
            Product("P010", "Chicken Breast (500g)", "meat", 8.90, 
                   {"calories": 165, "protein": 31, "sugar": 0, "fat": 4, "sodium": 74}, 340),
            Product("P011", "Salmon Fillet (400g)", "fish", 12.50, 
                   {"calories": 208, "protein": 22, "sugar": 0, "fat": 12, "omega3": 1.8}, 180),
            Product("P012", "Tofu (300g)", "protein", 3.80, 
                   {"calories": 94, "protein": 10, "sugar": 1, "fat": 5, "calcium": 350}, 150),
            
            # Beverages
            Product("P013", "Green Tea", "beverages", 3.50, 
                   {"calories": 2, "protein": 0, "sugar": 0, "antioxidants": 50}, 280),
            Product("P014", "Low-Sugar Orange Juice", "beverages", 4.20, 
                   {"calories": 110, "protein": 2, "sugar": 21, "vitamin_c": 124}, 220),
            
            # Snacks (healthier options)
            Product("P015", "Mixed Nuts (200g)", "snacks", 6.80, 
                   {"calories": 607, "protein": 20, "sugar": 4, "fiber": 9, "healthy_fats": 54}, 190),
            Product("P016", "Dark Chocolate (85%)", "snacks", 4.50, 
                   {"calories": 170, "protein": 2, "sugar": 7, "fiber": 3, "antioxidants": 40}, 120),
        ]
        
        for product in sample_products:
            self.products[product.product_id] = product
        
        self.save_products()

    def _build_category_map(self):
        """Build a map of categories to product IDs"""
        self.category_map = {}
        for product_id, product in self.products.items():
            if product.category not in self.category_map:
                self.category_map[product.category] = []
            self.category_map[product.category].append(product_id)

    def get_product(self, product_id: str) -> Optional[Product]:
        return self.products.get(product_id)

    def get_products_by_category(self, category: str) -> List[Product]:
        product_ids = self.category_map.get(category, [])
        return [self.products[pid] for pid in product_ids]

    def get_popular_products(self, limit: int = 10) -> List[Product]:
        """Get most popular products based on monthly sales"""
        sorted_products = sorted(self.products.values(), 
                               key=lambda p: p.monthly_sales, reverse=True)
        return sorted_products[:limit]

    def get_products_by_price_range(self, min_price: float, max_price: float) -> List[Product]:
        return [p for p in self.products.values() 
                if min_price <= p.price <= max_price]

    def get_healthier_alternatives(self, product_id: str, limit: int = 3) -> List[Product]:
        """Find healthier alternatives in the same category"""
        product = self.get_product(product_id)
        if not product:
            return []
        
        category_products = self.get_products_by_category(product.category)
        original_score = product.get_nutritional_score()
        
        # Filter products with better nutritional scores
        alternatives = [p for p in category_products 
                       if p.product_id != product_id and 
                       p.get_nutritional_score() > original_score]
        
        # Sort by nutritional score (descending)
        alternatives.sort(key=lambda p: p.get_nutritional_score(), reverse=True)
        
        return alternatives[:limit]

    def get_similar_products(self, product_id: str, limit: int = 5) -> List[Product]:
        """Find products in the same category with similar price range"""
        product = self.get_product(product_id)
        if not product:
            return []
        
        category_products = self.get_products_by_category(product.category)
        price_tolerance = 0.3  # 30% price tolerance
        
        similar = [p for p in category_products 
                  if p.product_id != product_id and 
                  abs(p.price - product.price) <= product.price * price_tolerance]
        
        # Sort by price similarity
        similar.sort(key=lambda p: abs(p.price - product.price))
        
        return similar[:limit]

    def search_products(self, query: str) -> List[Product]:
        """Search products by name or description"""
        query = query.lower()
        results = []
        
        for product in self.products.values():
            if (query in product.name.lower() or 
                query in product.description.lower() or
                query in product.category.lower()):
                results.append(product)
        
        return results

    def get_category_stats(self) -> Dict[str, Dict]:
        """Get statistics for each category"""
        stats = {}
        for category, product_ids in self.category_map.items():
            products = [self.products[pid] for pid in product_ids]
            stats[category] = {
                'count': len(products),
                'avg_price': sum(p.price for p in products) / len(products),
                'total_sales': sum(p.monthly_sales for p in products),
                'avg_nutritional_score': sum(p.get_nutritional_score() for p in products) / len(products)
            }
        return stats

    def update_sales(self, product_id: str, quantity: int = 1):
        """Update monthly sales for a product"""
        product = self.get_product(product_id)
        if product:
            product.monthly_sales += quantity
            self.save_products()