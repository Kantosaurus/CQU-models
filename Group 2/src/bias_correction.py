"""
Reviewer Bias Correction System
Group 2 - Chongqing Food Spice Prediction

This module implements advanced bias correction algorithms:
1. Individual reviewer bias detection and quantification
2. Temporal bias correction (rating drift over time)
3. Cultural bias adjustment (different regional tolerance baselines)
4. Systematic bias patterns identification
5. Confidence-weighted bias correction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import sqlite3
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BiasProfile:
    """Profile of a reviewer's biases"""
    user_id: str
    systematic_bias: float  # Overall tendency to over/under-rate
    cultural_bias: float    # Deviation from regional baseline
    temporal_trend: float   # Change in rating pattern over time
    consistency_score: float  # How consistent the user's ratings are
    confidence: float       # Confidence in bias measurements
    sample_size: int       # Number of ratings used for bias calculation
    last_updated: datetime
    
@dataclass 
class CorrectionResult:
    """Result of bias correction"""
    original_rating: float
    corrected_rating: float
    correction_applied: float
    confidence: float
    correction_reason: str

class ReviewerBiasCorrector:
    """Advanced system for detecting and correcting reviewer biases"""
    
    def __init__(self, db_path: str = "data/spice_database.db"):
        self.db_path = db_path
        self.setup_bias_tables()
        
        # Bias detection parameters
        self.min_ratings_for_bias_detection = 10
        self.temporal_window_days = 180
        self.statistical_significance_threshold = 0.05
        
        # Regional baselines for Chongqing
        self.regional_tolerance_baseline = {
            'Chongqing': 3.8,    # Higher baseline tolerance in Chongqing
            'Sichuan': 3.6,      # High tolerance in broader Sichuan
            'Other_China': 2.8,  # Lower baseline for other regions
            'International': 2.2  # Much lower baseline for international users
        }
        
        # Bias correction weights
        self.correction_weights = {
            'systematic_bias': 0.4,
            'cultural_bias': 0.3,
            'temporal_trend': 0.2,
            'consistency_adjustment': 0.1
        }
    
    def setup_bias_tables(self):
        """Setup database tables for bias tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS bias_profiles (
            user_id TEXT PRIMARY KEY,
            systematic_bias REAL,
            cultural_bias REAL,
            temporal_trend REAL,
            consistency_score REAL,
            confidence REAL,
            sample_size INTEGER,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS bias_corrections (
            correction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            dish_id TEXT,
            original_rating REAL,
            corrected_rating REAL,
            correction_applied REAL,
            confidence REAL,
            correction_reason TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS regional_baselines (
            region TEXT PRIMARY KEY,
            baseline_tolerance REAL,
            sample_size INTEGER,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def detect_systematic_bias(self, user_id: str) -> Tuple[float, float]:
        """Detect systematic bias in user's ratings compared to dish base levels"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT r.spice_rating, d.base_spice_level, r.timestamp
        FROM user_ratings r
        JOIN dishes d ON r.dish_id = d.dish_id
        WHERE r.user_id = ? AND d.base_spice_level IS NOT NULL
        ORDER BY r.timestamp DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=(user_id,))
        conn.close()
        
        if len(df) < self.min_ratings_for_bias_detection:
            return 0.0, 0.0  # No systematic bias detected, low confidence
        
        # Calculate differences between user ratings and base spice levels
        rating_differences = df['spice_rating'] - df['base_spice_level']
        
        # Statistical test for systematic bias
        t_stat, p_value = stats.ttest_1samp(rating_differences, 0)
        
        if p_value < self.statistical_significance_threshold:
            # Significant systematic bias detected
            systematic_bias = rating_differences.mean()
            confidence = min(0.95, (1 - p_value) * len(df) / 50)  # Scale confidence with sample size
        else:
            systematic_bias = 0.0
            confidence = 0.3
        
        return systematic_bias, confidence
    
    def detect_cultural_bias(self, user_id: str) -> Tuple[float, float]:
        """Detect cultural bias based on deviation from regional baseline"""
        # Get user's region (would be stored in user profile in real system)
        user_region = self._get_user_region(user_id)
        regional_baseline = self.regional_tolerance_baseline.get(user_region, 3.0)
        
        # Get user's average spice rating
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT AVG(spice_rating) as avg_rating, COUNT(*) as rating_count
        FROM user_ratings
        WHERE user_id = ?
        '''
        
        result = pd.read_sql_query(query, conn, params=(user_id,))
        conn.close()
        
        if result['rating_count'].iloc[0] < self.min_ratings_for_bias_detection:
            return 0.0, 0.0
        
        user_avg_rating = result['avg_rating'].iloc[0]
        cultural_bias = user_avg_rating - regional_baseline
        
        # Confidence based on sample size
        sample_size = result['rating_count'].iloc[0]
        confidence = min(0.9, sample_size / 30)
        
        return cultural_bias, confidence
    
    def detect_temporal_bias(self, user_id: str) -> Tuple[float, float]:
        """Detect temporal trends in user's rating patterns"""
        conn = sqlite3.connect(self.db_path)
        
        # Get ratings from recent temporal window
        cutoff_date = datetime.now() - timedelta(days=self.temporal_window_days)
        
        query = '''
        SELECT spice_rating, timestamp
        FROM user_ratings
        WHERE user_id = ? AND timestamp > ?
        ORDER BY timestamp ASC
        '''
        
        df = pd.read_sql_query(query, conn, params=(user_id, cutoff_date))
        conn.close()
        
        if len(df) < 15:  # Need minimum ratings for trend detection
            return 0.0, 0.0
        
        # Convert timestamps to days since first rating
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        first_rating_date = df['timestamp'].min()
        df['days_since_start'] = (df['timestamp'] - first_rating_date).dt.days
        
        # Linear regression to detect trend
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df['days_since_start'], df['spice_rating']
            )
            
            if p_value < self.statistical_significance_threshold and abs(r_value) > 0.3:
                temporal_trend = slope * 30  # Convert to monthly trend
                confidence = min(0.9, abs(r_value))
            else:
                temporal_trend = 0.0
                confidence = 0.2
                
        except Exception as e:
            logger.warning(f"Error calculating temporal trend for {user_id}: {e}")
            temporal_trend = 0.0
            confidence = 0.0
        
        return temporal_trend, confidence
    
    def calculate_consistency_score(self, user_id: str) -> Tuple[float, float]:
        """Calculate how consistent user's ratings are"""
        conn = sqlite3.connect(self.db_path)
        
        # Get ratings for same dishes (if any)
        query = '''
        SELECT dish_id, AVG(spice_rating) as avg_rating, 
               STDEV(spice_rating) as rating_std, COUNT(*) as rating_count
        FROM user_ratings
        WHERE user_id = ?
        GROUP BY dish_id
        HAVING COUNT(*) > 1
        '''
        
        df = pd.read_sql_query(query, conn, params=(user_id,))
        
        if df.empty:
            # Fallback: calculate overall rating variance
            query_all = '''
            SELECT spice_rating FROM user_ratings WHERE user_id = ?
            '''
            all_ratings = pd.read_sql_query(query_all, conn, params=(user_id,))
            conn.close()
            
            if len(all_ratings) < 5:
                return 0.5, 0.2  # Default moderate consistency, low confidence
            
            overall_std = all_ratings['spice_rating'].std()
            consistency_score = max(0, 1 - (overall_std / 2))  # Higher std = lower consistency
            confidence = min(0.7, len(all_ratings) / 20)
            
        else:
            conn.close()
            # Average consistency across dishes user rated multiple times
            avg_std = df['rating_std'].mean()
            consistency_score = max(0, 1 - (avg_std / 1.5))
            confidence = min(0.9, len(df) / 10)
        
        return consistency_score, confidence
    
    def build_bias_profile(self, user_id: str) -> BiasProfile:
        """Build comprehensive bias profile for user"""
        logger.info(f"Building bias profile for user {user_id}")
        
        # Detect different types of bias
        systematic_bias, sys_confidence = self.detect_systematic_bias(user_id)
        cultural_bias, cult_confidence = self.detect_cultural_bias(user_id)
        temporal_trend, temp_confidence = self.detect_temporal_bias(user_id)
        consistency_score, cons_confidence = self.calculate_consistency_score(user_id)
        
        # Calculate overall confidence
        confidences = [sys_confidence, cult_confidence, temp_confidence, cons_confidence]
        overall_confidence = np.mean([c for c in confidences if c > 0])
        
        # Get sample size
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM user_ratings WHERE user_id = ?', (user_id,))
        sample_size = cursor.fetchone()[0]
        conn.close()
        
        profile = BiasProfile(
            user_id=user_id,
            systematic_bias=systematic_bias,
            cultural_bias=cultural_bias,
            temporal_trend=temporal_trend,
            consistency_score=consistency_score,
            confidence=overall_confidence,
            sample_size=sample_size,
            last_updated=datetime.now()
        )
        
        # Store profile
        self._store_bias_profile(profile)
        
        return profile
    
    def apply_bias_correction(self, user_id: str, original_rating: float,
                            dish_id: str = None) -> CorrectionResult:
        """Apply bias correction to a rating"""
        # Get or build bias profile
        bias_profile = self.get_bias_profile(user_id)
        if not bias_profile:
            bias_profile = self.build_bias_profile(user_id)
        
        # Calculate total correction
        corrections = []
        
        # Systematic bias correction
        if abs(bias_profile.systematic_bias) > 0.2:
            systematic_correction = -bias_profile.systematic_bias * self.correction_weights['systematic_bias']
            corrections.append(('systematic', systematic_correction))
        
        # Cultural bias correction
        if abs(bias_profile.cultural_bias) > 0.3:
            cultural_correction = -bias_profile.cultural_bias * self.correction_weights['cultural_bias']
            corrections.append(('cultural', cultural_correction))
        
        # Temporal trend correction (recent trend gets more weight)
        if abs(bias_profile.temporal_trend) > 0.1:
            temporal_correction = -bias_profile.temporal_trend * self.correction_weights['temporal_trend']
            corrections.append(('temporal', temporal_correction))
        
        # Consistency adjustment
        if bias_profile.consistency_score < 0.6:
            # Less consistent users get their extreme ratings moderated
            center_rating = 3.0  # Middle of 1-5 scale
            consistency_correction = (center_rating - original_rating) * (1 - bias_profile.consistency_score) * self.correction_weights['consistency_adjustment']
            corrections.append(('consistency', consistency_correction))
        
        # Apply corrections
        total_correction = sum(correction for _, correction in corrections)
        
        # Weight by confidence
        confidence_weighted_correction = total_correction * bias_profile.confidence
        
        # Apply correction
        corrected_rating = original_rating + confidence_weighted_correction
        
        # Ensure rating stays within valid range
        corrected_rating = max(1, min(5, corrected_rating))
        
        # Generate explanation
        correction_reasons = []
        for correction_type, correction_value in corrections:
            if abs(correction_value) > 0.05:
                correction_reasons.append(f"{correction_type} bias ({correction_value:+.2f})")
        
        correction_reason = "; ".join(correction_reasons) if correction_reasons else "no significant bias detected"
        
        result = CorrectionResult(
            original_rating=original_rating,
            corrected_rating=corrected_rating,
            correction_applied=confidence_weighted_correction,
            confidence=bias_profile.confidence,
            correction_reason=correction_reason
        )
        
        # Store correction result
        self._store_correction_result(user_id, dish_id, result)
        
        return result
    
    def batch_correct_ratings(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Apply bias correction to a batch of ratings"""
        logger.info(f"Applying bias correction to {len(ratings_df)} ratings")
        
        corrected_ratings = []
        
        for _, row in ratings_df.iterrows():
            correction = self.apply_bias_correction(
                row['user_id'], 
                row['spice_rating'],
                row.get('dish_id', None)
            )
            
            corrected_ratings.append({
                'user_id': row['user_id'],
                'dish_id': row.get('dish_id', None),
                'original_rating': correction.original_rating,
                'corrected_rating': correction.corrected_rating,
                'correction_applied': correction.correction_applied,
                'confidence': correction.confidence
            })
        
        return pd.DataFrame(corrected_ratings)
    
    def get_bias_profile(self, user_id: str) -> Optional[BiasProfile]:
        """Retrieve bias profile from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM bias_profiles WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return BiasProfile(
                user_id=row[0],
                systematic_bias=row[1],
                cultural_bias=row[2],
                temporal_trend=row[3],
                consistency_score=row[4],
                confidence=row[5],
                sample_size=row[6],
                last_updated=datetime.fromisoformat(row[7])
            )
        
        return None
    
    def update_bias_profile(self, user_id: str):
        """Update bias profile with new data"""
        existing_profile = self.get_bias_profile(user_id)
        
        if not existing_profile or (datetime.now() - existing_profile.last_updated).days > 30:
            # Rebuild profile if it's old or doesn't exist
            self.build_bias_profile(user_id)
        else:
            # Incremental update (simplified for now)
            logger.info(f"Bias profile for {user_id} is recent, skipping update")
    
    def _get_user_region(self, user_id: str) -> str:
        """Get user's region (placeholder - would be in user profile)"""
        # In real implementation, this would look up user's location from profile
        # For now, assume most users are from Chongqing
        return 'Chongqing'
    
    def _store_bias_profile(self, profile: BiasProfile):
        """Store bias profile in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR REPLACE INTO bias_profiles
        (user_id, systematic_bias, cultural_bias, temporal_trend, 
         consistency_score, confidence, sample_size, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            profile.user_id, profile.systematic_bias, profile.cultural_bias,
            profile.temporal_trend, profile.consistency_score, profile.confidence,
            profile.sample_size, profile.last_updated
        ))
        
        conn.commit()
        conn.close()
    
    def _store_correction_result(self, user_id: str, dish_id: str, result: CorrectionResult):
        """Store bias correction result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO bias_corrections
        (user_id, dish_id, original_rating, corrected_rating, 
         correction_applied, confidence, correction_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, dish_id, result.original_rating, result.corrected_rating,
            result.correction_applied, result.confidence, result.correction_reason
        ))
        
        conn.commit()
        conn.close()
    
    def analyze_bias_patterns(self) -> Dict[str, Any]:
        """Analyze bias patterns across all users"""
        conn = sqlite3.connect(self.db_path)
        
        # Get all bias profiles
        profiles_df = pd.read_sql_query('SELECT * FROM bias_profiles', conn)
        
        if profiles_df.empty:
            conn.close()
            return {"error": "No bias profiles found"}
        
        analysis = {
            'total_users_analyzed': len(profiles_df),
            'avg_systematic_bias': profiles_df['systematic_bias'].mean(),
            'avg_cultural_bias': profiles_df['cultural_bias'].mean(),
            'avg_temporal_trend': profiles_df['temporal_trend'].mean(),
            'avg_consistency_score': profiles_df['consistency_score'].mean(),
            'high_bias_users': len(profiles_df[profiles_df['systematic_bias'].abs() > 0.5]),
            'low_consistency_users': len(profiles_df[profiles_df['consistency_score'] < 0.5])
        }
        
        # Bias distribution
        analysis['systematic_bias_distribution'] = {
            'very_negative': len(profiles_df[profiles_df['systematic_bias'] < -1.0]),
            'negative': len(profiles_df[(profiles_df['systematic_bias'] >= -1.0) & (profiles_df['systematic_bias'] < -0.3)]),
            'neutral': len(profiles_df[profiles_df['systematic_bias'].abs() <= 0.3]),
            'positive': len(profiles_df[(profiles_df['systematic_bias'] > 0.3) & (profiles_df['systematic_bias'] <= 1.0)]),
            'very_positive': len(profiles_df[profiles_df['systematic_bias'] > 1.0])
        }
        
        conn.close()
        return analysis
    
    def get_correction_statistics(self) -> Dict[str, Any]:
        """Get statistics about bias corrections applied"""
        conn = sqlite3.connect(self.db_path)
        
        corrections_df = pd.read_sql_query('SELECT * FROM bias_corrections', conn)
        conn.close()
        
        if corrections_df.empty:
            return {"error": "No corrections applied yet"}
        
        stats = {
            'total_corrections': len(corrections_df),
            'avg_correction_magnitude': corrections_df['correction_applied'].abs().mean(),
            'corrections_by_direction': {
                'increased_rating': len(corrections_df[corrections_df['correction_applied'] > 0]),
                'decreased_rating': len(corrections_df[corrections_df['correction_applied'] < 0]),
                'no_change': len(corrections_df[corrections_df['correction_applied'].abs() < 0.05])
            },
            'avg_confidence': corrections_df['confidence'].mean(),
            'significant_corrections': len(corrections_df[corrections_df['correction_applied'].abs() > 0.3])
        }
        
        return stats

if __name__ == "__main__":
    # Example usage
    corrector = ReviewerBiasCorrector()
    
    # Build bias profile for a user
    profile = corrector.build_bias_profile("U001")
    print(f"Bias profile for U001:")
    print(f"  Systematic bias: {profile.systematic_bias:.3f}")
    print(f"  Cultural bias: {profile.cultural_bias:.3f}")
    print(f"  Consistency: {profile.consistency_score:.3f}")
    print(f"  Confidence: {profile.confidence:.3f}")
    
    # Apply bias correction
    correction = corrector.apply_bias_correction("U001", 4.0, "CQ001")
    print(f"\nBias correction:")
    print(f"  Original rating: {correction.original_rating}")
    print(f"  Corrected rating: {correction.corrected_rating}")
    print(f"  Correction applied: {correction.correction_applied:+.3f}")
    print(f"  Reason: {correction.correction_reason}")
    
    # Analyze patterns
    analysis = corrector.analyze_bias_patterns()
    print(f"\nBias analysis: {analysis}")