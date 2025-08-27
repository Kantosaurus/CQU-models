# Smart Cart Robot Configuration Settings
# Group 5 - CQU AI Project

import os

# Base Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Database/Storage Settings
CUSTOMER_DATA_FILE = os.path.join(DATA_DIR, 'customers.pkl')
PRODUCT_DATA_FILE = os.path.join(DATA_DIR, 'products.pkl')

# Face Recognition Settings
FACE_RECOGNITION_CONFIG = {
    'model': 'hog',  # 'hog' for CPU, 'cnn' for GPU
    'tolerance': 0.6,
    'min_face_size': 50,
    'camera_index': 0,
    'frame_width': 640,
    'frame_height': 480,
    'fps': 30
}

# Voice Recognition Settings
VOICE_RECOGNITION_CONFIG = {
    'wake_words': ['hello', 'smart cart', 'help me', 'assistant'],
    'timeout_seconds': 10,
    'language': 'en-US',
    'enable_voice_commands': True
}

# Recommendation System Settings
RECOMMENDATION_CONFIG = {
    'max_recommendations': 10,
    'popularity_weight': 0.4,
    'content_weight': 0.6,
    'health_bonus_multiplier': 0.1,
    'price_tolerance': 0.3,
    'new_customer_threshold': 0  # Number of purchases to be considered "new"
}

# Elderly-Specific Preferences
ELDERLY_PREFERENCES = {
    'preferred_categories': {
        'dairy': 1.2,      # Good for bone health
        'fruits': 1.3,     # Vitamins and fiber
        'vegetables': 1.3, # Nutrients and fiber
        'fish': 1.4,       # Omega-3 for brain health
        'grains': 1.1,     # Energy and fiber
        'beverages': 0.9,  # Lower priority
        'protein': 1.2,    # Important for muscle health
        'meat': 1.0,       # Moderate priority
        'snacks': 0.7      # Lower priority for health
    },
    'max_budget_suggestion': 100.0,  # Default budget for shopping list suggestions
    'price_sensitivity': {
        'low': 5.0,   # Items under $5 get price bonus
        'medium': 10.0,  # Items under $10 are neutral
        'high': 20.0     # Items over $20 get price penalty
    }
}

# Health Score Configuration
HEALTH_SCORE_WEIGHTS = {
    'protein': 2.0,    # Higher protein is better
    'fiber': 1.5,      # Higher fiber is better
    'sugar': -0.5,     # Lower sugar is better
    'sodium': -0.001,  # Lower sodium is better (weight scaled for mg values)
    'calcium': 0.01,   # Higher calcium is better (weight scaled for mg values)
    'iron': 0.5,       # Higher iron is better
    'vitamin_c': 0.01, # Higher vitamin C is better
    'omega3': 1.0      # Higher omega-3 is better
}

# Shopping Cart Settings
CART_CONFIG = {
    'max_items_per_session': 50,
    'session_timeout_minutes': 60,
    'auto_save_interval_seconds': 30
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': os.path.join(BASE_DIR, 'logs', 'smart_cart.log'),
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(LOGGING_CONFIG['log_file']), exist_ok=True)

# UI/Display Settings (for future robot interface)
DISPLAY_CONFIG = {
    'screen_width': 1024,
    'screen_height': 768,
    'font_size_large': 24,
    'font_size_medium': 18,
    'font_size_small': 14,
    'colors': {
        'primary': '#2E7D32',      # Green
        'secondary': '#1976D2',    # Blue
        'success': '#4CAF50',      # Light Green
        'warning': '#FF9800',      # Orange
        'error': '#F44336',        # Red
        'background': '#FAFAFA',   # Light Gray
        'text': '#212121'          # Dark Gray
    }
}

# Chongqing-Specific Settings
LOCATION_CONFIG = {
    'city': 'Chongqing',
    'country': 'China',
    'currency': 'CNY',
    'language': 'zh-CN',
    'local_preferences': {
        'spicy_food_preference': 0.8,  # Chongqing loves spicy food
        'hot_pot_ingredients': 1.2,    # Boost for hot pot ingredients
        'local_vegetables': 1.1        # Boost for locally grown vegetables
    }
}

# Security Settings
SECURITY_CONFIG = {
    'face_data_encryption': False,  # Set to True in production
    'customer_data_retention_days': 365,
    'anonymous_mode': False,  # Allow shopping without face recognition
    'data_privacy_compliance': 'GDPR'  # or 'CCPA', 'local'
}

# Performance Settings
PERFORMANCE_CONFIG = {
    'cache_recommendations': True,
    'cache_timeout_seconds': 300,  # 5 minutes
    'max_concurrent_customers': 10,
    'background_processing': True
}

# Debug and Development Settings
DEBUG_CONFIG = {
    'debug_mode': True,
    'verbose_logging': True,
    'save_debug_images': False,
    'mock_camera': True,  # Use mock camera for development
    'test_data_enabled': True
}