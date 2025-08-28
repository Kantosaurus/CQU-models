"""
User Tolerance Profiling System
Group 2 - Chongqing Food Spice Prediction

This module manages user spice tolerance profiling:
1. Tolerance level assessment and calibration
2. Preference learning from rating patterns
3. Bias detection and correction
4. Dynamic profile updates
5. Tolerance building tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import sqlite3
import json
from collections import defaultdict
import logging
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class ToleranceProfile:
    """User's spice tolerance profile"""
    user_id: str
    tolerance_level: int  # 1-5 scale
    confidence_score: float  # How confident we are in this assessment
    preferred_spice_range: Tuple[int, int]  # (min, max) preferred levels
    cuisine_preferences: Dict[str, float]  # cuisine -> preference score
    rating_bias: float  # tendency to over/under-rate spice levels
    tolerance_trend: float  # increasing/decreasing tolerance over time
    last_updated: datetime
    calibration_data: Dict[str, float]  # calibration metrics

@dataclass  
class ToleranceAssessment:
    """Assessment of user's spice tolerance"""
    user_id: str
    assessed_tolerance: int
    confidence: float
    evidence_count: int
    assessment_method: str
    created_at: datetime

class UserProfilingSystem:
    """System for building and maintaining user spice tolerance profiles"""
    
    def __init__(self, db_path: str = "data/spice_database.db"):
        self.db_path = db_path
        self.setup_profiling_tables()
        
        # Tolerance assessment thresholds
        self.min_ratings_for_assessment = 5
        self.confidence_threshold = 0.7
        
        # Bias detection parameters
        self.bias_detection_window = 50  # number of ratings to consider
        self.significant_bias_threshold = 0.5  # bias score threshold
        
    def setup_profiling_tables(self):
        """Setup additional tables for user profiling"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tolerance_assessments (
            assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            assessed_tolerance INTEGER,
            confidence REAL,
            evidence_count INTEGER,
            assessment_method TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tolerance_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            tolerance_level INTEGER,
            timestamp TIMESTAMP,
            trigger_event TEXT  -- what caused the tolerance update
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS calibration_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            dish_id TEXT,
            predicted_spice REAL,
            actual_rating INTEGER,
            calibration_score REAL,
            session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def assess_user_tolerance(self, user_id: str) -> ToleranceAssessment:
        """Assess user's spice tolerance based on rating patterns"""
        ratings_data = self._get_user_rating_history(user_id)
        
        if len(ratings_data) < self.min_ratings_for_assessment:
            # Use default/survey-based assessment for new users
            return self._default_tolerance_assessment(user_id)
        
        # Method 1: Analyze rating patterns vs known dish spice levels
        tolerance_from_patterns = self._assess_from_rating_patterns(ratings_data)
        
        # Method 2: Compare with similar users (collaborative approach)
        tolerance_from_similarity = self._assess_from_user_similarity(user_id, ratings_data)
        
        # Method 3: Statistical analysis of spice ratings distribution
        tolerance_from_distribution = self._assess_from_rating_distribution(ratings_data)
        
        # Combine assessments with weighted average
        methods = [
            (tolerance_from_patterns, 0.5),
            (tolerance_from_similarity, 0.3), 
            (tolerance_from_distribution, 0.2)
        ]
        
        weighted_tolerance = sum(tolerance * weight for (tolerance, conf), weight in methods)
        avg_confidence = np.mean([conf for (tolerance, conf), weight in methods])
        
        final_tolerance = max(1, min(5, round(weighted_tolerance)))
        
        assessment = ToleranceAssessment(
            user_id=user_id,
            assessed_tolerance=final_tolerance,
            confidence=avg_confidence,
            evidence_count=len(ratings_data),
            assessment_method="combined_analysis",
            created_at=datetime.now()
        )
        
        # Store assessment
        self._store_tolerance_assessment(assessment)
        
        return assessment
    
    def _get_user_rating_history(self, user_id: str) -> pd.DataFrame:
        """Get user's rating history with dish information"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT 
            r.spice_rating,
            r.overall_rating,
            r.timestamp,
            d.base_spice_level,
            d.lab_scoville,
            d.cuisine_type,
            d.cooking_method
        FROM user_ratings r
        JOIN dishes d ON r.dish_id = d.dish_id
        WHERE r.user_id = ?
        ORDER BY r.timestamp ASC
        '''
        
        df = pd.read_sql_query(query, conn, params=(user_id,))
        conn.close()
        
        return df
    
    def _assess_from_rating_patterns(self, ratings_data: pd.DataFrame) -> Tuple[float, float]:
        """Assess tolerance based on how user rates dishes vs their known spice levels"""
        if ratings_data.empty or ratings_data['base_spice_level'].isna().all():
            return 3.0, 0.3  # default moderate tolerance, low confidence
        
        # Remove entries without base spice level data
        clean_data = ratings_data.dropna(subset=['base_spice_level'])
        
        if len(clean_data) < 3:
            return 3.0, 0.3
        
        # Calculate the difference between user's rating and dish's actual spice level
        spice_perception_diff = clean_data['spice_rating'] - clean_data['base_spice_level']
        
        # Users with higher tolerance tend to rate spicy dishes as less spicy
        # If user consistently rates spicy dishes lower, they have higher tolerance
        avg_diff = spice_perception_diff.mean()
        
        # Convert difference to tolerance scale (inverse relationship)
        if avg_diff <= -1.5:
            assessed_tolerance = 5  # Very high tolerance
        elif avg_diff <= -0.5:
            assessed_tolerance = 4  # High tolerance
        elif avg_diff <= 0.5:
            assessed_tolerance = 3  # Medium tolerance
        elif avg_diff <= 1.5:
            assessed_tolerance = 2  # Low tolerance
        else:
            assessed_tolerance = 1  # Very low tolerance
        
        # Confidence based on consistency of ratings
        consistency = 1 / (1 + spice_perception_diff.std())
        confidence = min(0.95, max(0.3, consistency))
        
        return float(assessed_tolerance), confidence
    
    def _assess_from_user_similarity(self, user_id: str, ratings_data: pd.DataFrame) -> Tuple[float, float]:
        """Assess tolerance by finding similar users and using their tolerance levels"""
        # Get ratings from all users for dishes this user has also rated
        conn = sqlite3.connect(self.db_path)
        
        user_dish_ids = self._get_user_rated_dishes(user_id)
        if not user_dish_ids:
            return 3.0, 0.2
        
        # Get other users who rated the same dishes
        dish_placeholders = ','.join(['?' for _ in user_dish_ids])
        query = f'''
        SELECT user_id, dish_id, spice_rating, user_tolerance_level
        FROM user_ratings 
        WHERE dish_id IN ({dish_placeholders}) AND user_id != ?
        AND user_tolerance_level IS NOT NULL
        '''
        
        params = list(user_dish_ids) + [user_id]
        similar_users_data = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if similar_users_data.empty:
            return 3.0, 0.2
        
        # Calculate similarity with other users based on rating patterns
        current_user_ratings = ratings_data.set_index(ratings_data.index)['spice_rating']
        
        user_similarities = []
        for other_user_id in similar_users_data['user_id'].unique():
            other_user_data = similar_users_data[similar_users_data['user_id'] == other_user_id]
            
            # Find common dishes
            common_dishes = set(ratings_data.index) & set(other_user_data.index)
            if len(common_dishes) >= 2:
                # Calculate rating correlation
                try:
                    correlation = np.corrcoef(
                        [current_user_ratings.loc[i] for i in common_dishes],
                        [other_user_data[other_user_data.index.isin(common_dishes)]['spice_rating'].iloc[0]]
                    )[0,1]
                    
                    if not np.isnan(correlation):
                        tolerance_level = other_user_data['user_tolerance_level'].iloc[0]
                        user_similarities.append((other_user_id, correlation, tolerance_level))
                except:
                    continue
        
        if not user_similarities:
            return 3.0, 0.2
        
        # Weight similar users by their correlation with current user
        total_weight = 0
        weighted_tolerance = 0
        
        for other_user, correlation, tolerance in user_similarities:
            if correlation > 0.3:  # Only consider positively correlated users
                weight = correlation ** 2  # Square to emphasize highly similar users
                weighted_tolerance += tolerance * weight
                total_weight += weight
        
        if total_weight > 0:
            estimated_tolerance = weighted_tolerance / total_weight
            confidence = min(0.8, total_weight / len(user_similarities))
        else:
            estimated_tolerance = 3.0
            confidence = 0.2
        
        return estimated_tolerance, confidence
    
    def _assess_from_rating_distribution(self, ratings_data: pd.DataFrame) -> Tuple[float, float]:
        """Assess tolerance from distribution of spice ratings given by user"""
        spice_ratings = ratings_data['spice_rating'].values
        
        if len(spice_ratings) < 3:
            return 3.0, 0.2
        
        # Calculate statistics of spice ratings
        mean_rating = np.mean(spice_ratings)
        std_rating = np.std(spice_ratings)
        
        # Users with higher tolerance give lower spice ratings on average
        # and have wider distribution (comfortable with range of spice levels)
        
        # Map mean rating to tolerance level (inverse relationship)
        if mean_rating <= 2.0:
            base_tolerance = 5
        elif mean_rating <= 2.5:
            base_tolerance = 4
        elif mean_rating <= 3.5:
            base_tolerance = 3
        elif mean_rating <= 4.0:
            base_tolerance = 2
        else:
            base_tolerance = 1
        
        # Adjust based on rating spread (higher std suggests higher tolerance)
        if std_rating > 1.2:
            tolerance_adjustment = 0.5
        elif std_rating < 0.6:
            tolerance_adjustment = -0.5
        else:
            tolerance_adjustment = 0
        
        estimated_tolerance = max(1, min(5, base_tolerance + tolerance_adjustment))
        
        # Confidence based on number of ratings and consistency
        sample_confidence = min(0.9, len(spice_ratings) / 20)
        consistency_confidence = 1 / (1 + std_rating)
        confidence = (sample_confidence + consistency_confidence) / 2
        
        return estimated_tolerance, confidence
    
    def _default_tolerance_assessment(self, user_id: str) -> ToleranceAssessment:
        """Default assessment for new users with insufficient data"""
        return ToleranceAssessment(
            user_id=user_id,
            assessed_tolerance=3,  # Assume moderate tolerance
            confidence=0.3,
            evidence_count=0,
            assessment_method="default",
            created_at=datetime.now()
        )
    
    def _get_user_rated_dishes(self, user_id: str) -> List[str]:
        """Get list of dishes rated by user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT DISTINCT dish_id FROM user_ratings WHERE user_id = ?', (user_id,))
        dish_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return dish_ids
    
    def detect_rating_bias(self, user_id: str) -> float:
        """Detect if user has systematic bias in spice ratings"""
        ratings_data = self._get_user_rating_history(user_id)
        
        if len(ratings_data) < self.bias_detection_window:
            return 0.0  # Insufficient data for bias detection
        
        # Take most recent ratings
        recent_ratings = ratings_data.tail(self.bias_detection_window)
        
        if recent_ratings['base_spice_level'].isna().all():
            return 0.0
        
        clean_data = recent_ratings.dropna(subset=['base_spice_level'])
        
        if len(clean_data) < 10:
            return 0.0
        
        # Calculate systematic difference between user's ratings and dish base levels
        rating_differences = clean_data['spice_rating'] - clean_data['base_spice_level']
        
        # Test if the mean difference is significantly different from 0
        t_stat, p_value = stats.ttest_1samp(rating_differences, 0)
        
        if p_value < 0.05:  # Significant bias detected
            bias_magnitude = rating_differences.mean()
            return bias_magnitude
        
        return 0.0
    
    def build_complete_profile(self, user_id: str) -> ToleranceProfile:
        """Build complete tolerance profile for user"""
        # Get tolerance assessment
        assessment = self.assess_user_tolerance(user_id)
        
        # Detect rating bias
        bias = self.detect_rating_bias(user_id)
        
        # Get preference analysis
        preferences = self._analyze_cuisine_preferences(user_id)
        
        # Calculate tolerance trend
        trend = self._calculate_tolerance_trend(user_id)
        
        # Determine preferred spice range
        spice_range = self._determine_preferred_spice_range(user_id, assessment.assessed_tolerance)
        
        profile = ToleranceProfile(
            user_id=user_id,
            tolerance_level=assessment.assessed_tolerance,
            confidence_score=assessment.confidence,
            preferred_spice_range=spice_range,
            cuisine_preferences=preferences,
            rating_bias=bias,
            tolerance_trend=trend,
            last_updated=datetime.now(),
            calibration_data={}
        )
        
        # Store profile in database
        self._store_tolerance_profile(profile)
        
        return profile
    
    def _analyze_cuisine_preferences(self, user_id: str) -> Dict[str, float]:
        """Analyze user's cuisine preferences based on ratings"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT d.cuisine_type, AVG(r.overall_rating) as avg_rating, COUNT(*) as rating_count
        FROM user_ratings r
        JOIN dishes d ON r.dish_id = d.dish_id
        WHERE r.user_id = ?
        GROUP BY d.cuisine_type
        HAVING COUNT(*) >= 2
        '''
        
        cuisine_data = pd.read_sql_query(query, conn, params=(user_id,))
        conn.close()
        
        if cuisine_data.empty:
            return {"Sichuan": 0.5, "Chongqing": 0.5}
        
        # Normalize preferences to 0-1 scale
        max_rating = cuisine_data['avg_rating'].max()
        min_rating = cuisine_data['avg_rating'].min()
        
        if max_rating == min_rating:
            # All cuisines rated equally
            preferences = {row['cuisine_type']: 0.5 for _, row in cuisine_data.iterrows()}
        else:
            preferences = {}
            for _, row in cuisine_data.iterrows():
                normalized_pref = (row['avg_rating'] - min_rating) / (max_rating - min_rating)
                preferences[row['cuisine_type']] = normalized_pref
        
        return preferences
    
    def _calculate_tolerance_trend(self, user_id: str) -> float:
        """Calculate if user's tolerance is increasing or decreasing over time"""
        ratings_data = self._get_user_rating_history(user_id)
        
        if len(ratings_data) < 10:
            return 0.0  # Insufficient data
        
        # Use spice ratings over time to detect trend
        ratings_data['timestamp'] = pd.to_datetime(ratings_data['timestamp'])
        ratings_data = ratings_data.sort_values('timestamp')
        
        # Calculate rolling average of spice ratings
        window_size = min(5, len(ratings_data) // 3)
        rolling_avg = ratings_data['spice_rating'].rolling(window=window_size).mean()
        
        if len(rolling_avg.dropna()) < 2:
            return 0.0
        
        # Linear regression on rolling averages to detect trend
        x = np.arange(len(rolling_avg.dropna()))
        y = rolling_avg.dropna().values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Return slope if trend is significant, otherwise 0
        if p_value < 0.05 and abs(r_value) > 0.3:
            return -slope  # Negative slope means increasing tolerance (lower ratings over time)
        
        return 0.0
    
    def _determine_preferred_spice_range(self, user_id: str, tolerance_level: int) -> Tuple[int, int]:
        """Determine user's preferred spice level range"""
        ratings_data = self._get_user_rating_history(user_id)
        
        if ratings_data.empty:
            # Default ranges based on tolerance level
            range_map = {
                1: (1, 2),
                2: (1, 3), 
                3: (2, 4),
                4: (3, 5),
                5: (4, 5)
            }
            return range_map.get(tolerance_level, (2, 4))
        
        # Find spice levels with highest overall ratings
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT d.base_spice_level, AVG(r.overall_rating) as avg_rating
        FROM user_ratings r
        JOIN dishes d ON r.dish_id = d.dish_id
        WHERE r.user_id = ? AND d.base_spice_level IS NOT NULL
        GROUP BY d.base_spice_level
        HAVING COUNT(*) >= 2
        '''
        
        spice_satisfaction = pd.read_sql_query(query, conn, params=(user_id,))
        conn.close()
        
        if spice_satisfaction.empty:
            range_map = {1: (1, 2), 2: (1, 3), 3: (2, 4), 4: (3, 5), 5: (4, 5)}
            return range_map.get(tolerance_level, (2, 4))
        
        # Find spice levels with ratings above average
        avg_satisfaction = spice_satisfaction['avg_rating'].mean()
        preferred_levels = spice_satisfaction[
            spice_satisfaction['avg_rating'] >= avg_satisfaction
        ]['base_spice_level'].values
        
        if len(preferred_levels) == 0:
            range_map = {1: (1, 2), 2: (1, 3), 3: (2, 4), 4: (3, 5), 5: (4, 5)}
            return range_map.get(tolerance_level, (2, 4))
        
        min_preferred = int(min(preferred_levels))
        max_preferred = int(max(preferred_levels))
        
        return (min_preferred, max_preferred)
    
    def _store_tolerance_assessment(self, assessment: ToleranceAssessment):
        """Store tolerance assessment in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO tolerance_assessments 
        (user_id, assessed_tolerance, confidence, evidence_count, assessment_method)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            assessment.user_id, assessment.assessed_tolerance, assessment.confidence,
            assessment.evidence_count, assessment.assessment_method
        ))
        
        conn.commit()
        conn.close()
    
    def _store_tolerance_profile(self, profile: ToleranceProfile):
        """Store complete tolerance profile in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR REPLACE INTO user_profiles
        (user_id, tolerance_level, preferred_min_spice, preferred_max_spice, 
         cuisine_preferences, bias_factor, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            profile.user_id, profile.tolerance_level,
            profile.preferred_spice_range[0], profile.preferred_spice_range[1],
            json.dumps(profile.cuisine_preferences), profile.rating_bias,
            profile.last_updated
        ))
        
        conn.commit()
        conn.close()
    
    def get_user_profile(self, user_id: str) -> Optional[ToleranceProfile]:
        """Retrieve user tolerance profile from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return ToleranceProfile(
                user_id=row[0],
                tolerance_level=row[1],
                confidence_score=0.8,  # Default, would track separately
                preferred_spice_range=(row[2], row[3]),
                cuisine_preferences=json.loads(row[4]),
                rating_bias=row[5],
                tolerance_trend=0.0,  # Would calculate dynamically
                last_updated=datetime.fromisoformat(row[6]),
                calibration_data={}
            )
        
        return None
    
    def update_profile_with_new_rating(self, user_id: str, dish_id: str, 
                                     spice_rating: int, overall_rating: float):
        """Update user profile when they provide a new rating"""
        # Check if profile needs updating (significant new data)
        profile = self.get_user_profile(user_id)
        
        if not profile or (datetime.now() - profile.last_updated).days > 30:
            # Rebuild complete profile
            self.build_complete_profile(user_id)
            logger.info(f"Rebuilt profile for user {user_id}")
        else:
            # Incremental update
            self._incremental_profile_update(user_id, dish_id, spice_rating, overall_rating)

    def _incremental_profile_update(self, user_id: str, dish_id: str, 
                                  spice_rating: int, overall_rating: float):
        """Perform incremental update to user profile"""
        # This would implement efficient incremental learning
        # For now, we'll trigger a full rebuild if needed
        pass

if __name__ == "__main__":
    # Example usage
    profiler = UserProfilingSystem()
    
    # Assess a user's tolerance
    assessment = profiler.assess_user_tolerance("U001")
    print(f"User U001 tolerance: {assessment.assessed_tolerance} (confidence: {assessment.confidence:.2f})")
    
    # Build complete profile
    profile = profiler.build_complete_profile("U001")
    print(f"Complete profile - Tolerance: {profile.tolerance_level}, Bias: {profile.rating_bias:.2f}")
    print(f"Preferences: {profile.cuisine_preferences}")
    print(f"Preferred spice range: {profile.preferred_spice_range}")