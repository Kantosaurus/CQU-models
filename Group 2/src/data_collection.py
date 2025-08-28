"""
Spice Level Data Collection System
Group 2 - Chongqing Food Spice Prediction

This module handles data collection from multiple sources:
1. Laboratory Scoville measurements
2. User spice tolerance ratings
3. Dish characteristics and ingredients
4. Restaurant-specific variations
5. Meituan review data integration
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DishData:
    """Data structure for dish information"""
    dish_id: str
    name: str
    restaurant_id: str
    cuisine_type: str
    main_ingredients: List[str]
    cooking_method: str
    spice_ingredients: List[str]  # peppers, spices used
    lab_scoville: Optional[float] = None
    base_spice_level: Optional[int] = None  # 1-5 scale
    price: Optional[float] = None
    description: str = ""
    
@dataclass
class UserRating:
    """User spice rating data"""
    user_id: str
    dish_id: str
    spice_rating: int  # 1-5 scale
    overall_rating: float  # 1-5 scale
    timestamp: datetime
    review_text: Optional[str] = None
    user_tolerance_level: Optional[int] = None

@dataclass
class UserProfile:
    """User spice tolerance profile"""
    user_id: str
    tolerance_level: int  # 1-5 scale (1=very low, 5=very high)
    preferred_spice_range: Tuple[int, int]  # (min, max) preferred spice levels
    cuisine_preferences: Dict[str, float]  # cuisine type -> preference score
    rating_history: List[UserRating]
    bias_factor: float = 0.0  # tendency to over/under-rate spice levels

class SpiceDataCollector:
    """Main data collection system for spice levels"""
    
    def __init__(self, db_path: str = "data/spice_database.db"):
        self.db_path = db_path
        self.setup_database()
        
        # Chongqing-specific spice ingredients and their typical Scoville ranges
        self.chongqing_spice_map = {
            'tian_jiao': (30000, 50000),      # 天椒 - Sky pepper
            'xiao_mi_la': (50000, 100000),    # 小米辣 - Small chili
            'er_jin_tiao': (15000, 30000),    # 二荆条 - Er Jing Tiao pepper
            'dou_ban_jiang': (5000, 15000),   # 豆瓣酱 - Doubanjiang
            'hua_jiao': (0, 100),             # 花椒 - Sichuan peppercorn (numbing, not hot)
            'gan_la_jiao': (25000, 40000),    # 干辣椒 - Dried chili
            'chao_tian_jiao': (100000, 350000), # 朝天椒 - Facing heaven pepper
        }
        
        # Cooking methods that affect spice intensity
        self.cooking_intensity_multipliers = {
            'stir_fry': 1.0,
            'deep_fry': 0.8,      # Oil reduces perceived heat
            'boil': 0.7,          # Water dilutes spice
            'steam': 0.6,         # Steam cooking is milder
            'grill': 1.2,         # Grilling intensifies flavors
            'hot_pot': 1.5,       # Hot pot cooking intensifies spice
            'dry_pot': 1.8,       # 干锅 - Very spicy cooking method
        }
    
    def setup_database(self):
        """Initialize SQLite database for storing spice data"""
        Path("data").mkdir(exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS dishes (
            dish_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            restaurant_id TEXT,
            cuisine_type TEXT,
            main_ingredients TEXT,  -- JSON array
            cooking_method TEXT,
            spice_ingredients TEXT, -- JSON array
            lab_scoville REAL,
            base_spice_level INTEGER,
            price REAL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_ratings (
            rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            dish_id TEXT,
            spice_rating INTEGER,
            overall_rating REAL,
            timestamp TIMESTAMP,
            review_text TEXT,
            user_tolerance_level INTEGER,
            FOREIGN KEY (dish_id) REFERENCES dishes (dish_id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            tolerance_level INTEGER,
            preferred_min_spice INTEGER,
            preferred_max_spice INTEGER,
            cuisine_preferences TEXT, -- JSON
            bias_factor REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS scoville_measurements (
            measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dish_id TEXT,
            scoville_value REAL,
            measurement_date TIMESTAMP,
            lab_name TEXT,
            confidence_score REAL,
            FOREIGN KEY (dish_id) REFERENCES dishes (dish_id)
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def add_dish(self, dish: DishData):
        """Add dish data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR REPLACE INTO dishes 
        (dish_id, name, restaurant_id, cuisine_type, main_ingredients, 
         cooking_method, spice_ingredients, lab_scoville, base_spice_level, price, description)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            dish.dish_id, dish.name, dish.restaurant_id, dish.cuisine_type,
            json.dumps(dish.main_ingredients), dish.cooking_method,
            json.dumps(dish.spice_ingredients), dish.lab_scoville,
            dish.base_spice_level, dish.price, dish.description
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Added dish: {dish.name}")
    
    def add_user_rating(self, rating: UserRating):
        """Add user rating to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO user_ratings 
        (user_id, dish_id, spice_rating, overall_rating, timestamp, review_text, user_tolerance_level)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            rating.user_id, rating.dish_id, rating.spice_rating, 
            rating.overall_rating, rating.timestamp, rating.review_text,
            rating.user_tolerance_level
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Added rating from user {rating.user_id} for dish {rating.dish_id}")
    
    def estimate_scoville_from_ingredients(self, spice_ingredients: List[str], 
                                         cooking_method: str) -> float:
        """Estimate Scoville rating based on ingredients and cooking method"""
        total_scoville = 0
        
        for ingredient in spice_ingredients:
            ingredient_lower = ingredient.lower()
            for spice_name, (min_scov, max_scov) in self.chongqing_spice_map.items():
                if spice_name in ingredient_lower or ingredient_lower in spice_name:
                    # Use average of range
                    avg_scoville = (min_scov + max_scov) / 2
                    total_scoville += avg_scoville
                    break
        
        # Apply cooking method multiplier
        multiplier = self.cooking_intensity_multipliers.get(cooking_method, 1.0)
        estimated_scoville = total_scoville * multiplier
        
        return max(0, estimated_scoville)
    
    def convert_scoville_to_scale(self, scoville: float) -> int:
        """Convert Scoville units to 1-5 spice level scale"""
        if scoville == 0:
            return 1
        elif scoville < 1000:
            return 1  # Mild
        elif scoville < 5000:
            return 2  # Medium-mild
        elif scoville < 25000:
            return 3  # Medium
        elif scoville < 75000:
            return 4  # Hot
        else:
            return 5  # Very hot
    
    def generate_sample_chongqing_dishes(self) -> List[DishData]:
        """Generate sample Chongqing dish data for testing"""
        sample_dishes = [
            DishData(
                dish_id="CQ001",
                name="麻婆豆腐 (Mapo Tofu)",
                restaurant_id="R001",
                cuisine_type="Sichuan",
                main_ingredients=["tofu", "ground_pork", "scallions"],
                cooking_method="stir_fry",
                spice_ingredients=["dou_ban_jiang", "hua_jiao", "gan_la_jiao"],
                price=28.0,
                description="Classic Sichuan dish with spicy and numbing flavors"
            ),
            DishData(
                dish_id="CQ002", 
                name="重庆火锅 (Chongqing Hot Pot)",
                restaurant_id="R002",
                cuisine_type="Chongqing",
                main_ingredients=["beef", "lamb", "vegetables", "noodles"],
                cooking_method="hot_pot",
                spice_ingredients=["chao_tian_jiao", "hua_jiao", "dou_ban_jiang", "xiao_mi_la"],
                price=88.0,
                description="Authentic Chongqing hot pot with numbing spice"
            ),
            DishData(
                dish_id="CQ003",
                name="口水鸡 (Saliva Chicken)",
                restaurant_id="R001",
                cuisine_type="Sichuan",
                main_ingredients=["chicken", "peanuts", "cucumber"],
                cooking_method="steam",
                spice_ingredients=["gan_la_jiao", "hua_jiao"],
                price=32.0,
                description="Cold chicken dish with spicy Sichuan sauce"
            ),
            DishData(
                dish_id="CQ004",
                name="辣子鸡 (Spicy Chicken)",
                restaurant_id="R003",
                cuisine_type="Chongqing",
                main_ingredients=["chicken", "dried_chilies", "peanuts"],
                cooking_method="stir_fry",
                spice_ingredients=["chao_tian_jiao", "er_jin_tiao"],
                price=45.0,
                description="Dry-fried chicken with tons of dried chilies"
            ),
            DishData(
                dish_id="CQ005",
                name="酸辣粉 (Hot and Sour Noodles)",
                restaurant_id="R004",
                cuisine_type="Chongqing",
                main_ingredients=["sweet_potato_noodles", "pork", "peanuts"],
                cooking_method="boil",
                spice_ingredients=["xiao_mi_la", "gan_la_jiao"],
                price=15.0,
                description="Street food noodles with spicy and sour flavors"
            ),
            DishData(
                dish_id="CQ006",
                name="水煮鱼 (Boiled Fish in Chili Oil)",
                restaurant_id="R002",
                cuisine_type="Sichuan",
                main_ingredients=["fish", "bean_sprouts", "cabbage"],
                cooking_method="boil",
                spice_ingredients=["dou_ban_jiang", "chao_tian_jiao", "hua_jiao"],
                price=68.0,
                description="Fish cooked in spicy oil with vegetables"
            ),
            DishData(
                dish_id="CQ007",
                name="干锅花菜 (Dry Pot Cauliflower)",
                restaurant_id="R003",
                cuisine_type="Chongqing",
                main_ingredients=["cauliflower", "pork_belly", "garlic"],
                cooking_method="dry_pot",
                spice_ingredients=["er_jin_tiao", "hua_jiao", "tian_jiao"],
                price=35.0,
                description="Spicy dry pot style cauliflower"
            ),
            DishData(
                dish_id="CQ008",
                name="担担面 (Dan Dan Noodles)",
                restaurant_id="R001",
                cuisine_type="Sichuan",
                main_ingredients=["noodles", "ground_pork", "preserved_vegetables"],
                cooking_method="boil",
                spice_ingredients=["hua_jiao", "gan_la_jiao"],
                price=22.0,
                description="Classic Sichuan noodles with sesame and spicy sauce"
            )
        ]
        
        # Calculate estimated Scoville and base spice levels
        for dish in sample_dishes:
            estimated_scoville = self.estimate_scoville_from_ingredients(
                dish.spice_ingredients, dish.cooking_method)
            dish.lab_scoville = estimated_scoville
            dish.base_spice_level = self.convert_scoville_to_scale(estimated_scoville)
        
        return sample_dishes
    
    def generate_sample_user_ratings(self, num_users: int = 100) -> List[UserRating]:
        """Generate sample user ratings for testing"""
        ratings = []
        dish_ids = ["CQ001", "CQ002", "CQ003", "CQ004", "CQ005", "CQ006", "CQ007", "CQ008"]
        
        # Generate user tolerance levels (1-5 scale)
        user_tolerances = {}
        for i in range(1, num_users + 1):
            user_id = f"U{i:03d}"
            # Normal distribution centered around 3, with some variation
            tolerance = max(1, min(5, int(np.random.normal(3, 1))))
            user_tolerances[user_id] = tolerance
        
        # Generate ratings based on user tolerance vs dish spice level
        for user_id, tolerance in user_tolerances.items():
            # Each user rates 3-8 dishes
            num_ratings = np.random.randint(3, 9)
            rated_dishes = np.random.choice(dish_ids, num_ratings, replace=False)
            
            for dish_id in rated_dishes:
                # Get dish base spice level (we'll need to look this up)
                dish_base_spice = 3  # Default, would be looked up in real implementation
                
                # Calculate spice rating based on user tolerance vs dish spice
                # Users with high tolerance rate spicy dishes lower, and vice versa
                tolerance_diff = tolerance - dish_base_spice
                
                if tolerance_diff >= 2:
                    # Dish is much less spicy than user's tolerance
                    spice_rating = max(1, dish_base_spice - 1)
                elif tolerance_diff >= 0:
                    # Dish matches user's tolerance or slightly below
                    spice_rating = dish_base_spice
                elif tolerance_diff >= -1:
                    # Dish is slightly spicier than user's tolerance
                    spice_rating = min(5, dish_base_spice + 1)
                else:
                    # Dish is much spicier than user's tolerance
                    spice_rating = 5
                
                # Add some random variation
                spice_rating = max(1, min(5, spice_rating + np.random.randint(-1, 2)))
                
                # Overall rating correlates with how well the spice level matches preference
                if abs(tolerance_diff) <= 1:
                    overall_rating = np.random.uniform(3.5, 5.0)
                else:
                    overall_rating = np.random.uniform(2.0, 4.0)
                
                rating = UserRating(
                    user_id=user_id,
                    dish_id=dish_id,
                    spice_rating=spice_rating,
                    overall_rating=overall_rating,
                    timestamp=datetime.now() - timedelta(days=np.random.randint(0, 180)),
                    user_tolerance_level=tolerance
                )
                ratings.append(rating)
        
        return ratings
    
    def initialize_sample_data(self):
        """Initialize database with sample Chongqing food data"""
        logger.info("Initializing sample data...")
        
        # Add sample dishes
        dishes = self.generate_sample_chongqing_dishes()
        for dish in dishes:
            self.add_dish(dish)
        
        # Add sample user ratings
        ratings = self.generate_sample_user_ratings()
        for rating in ratings:
            self.add_user_rating(rating)
        
        logger.info(f"Sample data initialized: {len(dishes)} dishes, {len(ratings)} ratings")
    
    def get_dish_data(self, dish_id: str) -> Optional[DishData]:
        """Retrieve dish data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM dishes WHERE dish_id = ?', (dish_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return DishData(
                dish_id=row[0],
                name=row[1],
                restaurant_id=row[2],
                cuisine_type=row[3],
                main_ingredients=json.loads(row[4]),
                cooking_method=row[5],
                spice_ingredients=json.loads(row[6]),
                lab_scoville=row[7],
                base_spice_level=row[8],
                price=row[9],
                description=row[10]
            )
        return None
    
    def get_user_ratings_for_dish(self, dish_id: str) -> List[UserRating]:
        """Get all user ratings for a specific dish"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT user_id, dish_id, spice_rating, overall_rating, timestamp, 
               review_text, user_tolerance_level
        FROM user_ratings WHERE dish_id = ?
        ''', (dish_id,))
        
        ratings = []
        for row in cursor.fetchall():
            rating = UserRating(
                user_id=row[0],
                dish_id=row[1], 
                spice_rating=row[2],
                overall_rating=row[3],
                timestamp=datetime.fromisoformat(row[4]),
                review_text=row[5],
                user_tolerance_level=row[6]
            )
            ratings.append(rating)
        
        conn.close()
        return ratings
    
    def export_training_data(self) -> pd.DataFrame:
        """Export data in format suitable for ML training"""
        conn = sqlite3.connect(self.db_path)
        
        # Join dishes and ratings data
        query = '''
        SELECT 
            r.user_id,
            r.dish_id,
            r.spice_rating as target_spice_level,
            r.overall_rating,
            r.user_tolerance_level,
            d.name as dish_name,
            d.restaurant_id,
            d.cuisine_type,
            d.main_ingredients,
            d.cooking_method,
            d.spice_ingredients,
            d.lab_scoville,
            d.base_spice_level,
            d.price
        FROM user_ratings r
        JOIN dishes d ON r.dish_id = d.dish_id
        ORDER BY r.timestamp DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Process JSON columns
        df['main_ingredients'] = df['main_ingredients'].apply(json.loads)
        df['spice_ingredients'] = df['spice_ingredients'].apply(json.loads)
        
        logger.info(f"Exported {len(df)} training samples")
        return df

if __name__ == "__main__":
    # Example usage
    collector = SpiceDataCollector()
    collector.initialize_sample_data()
    
    # Export training data
    training_data = collector.export_training_data()
    print(f"Training data shape: {training_data.shape}")
    print("\nSample data:")
    print(training_data.head())