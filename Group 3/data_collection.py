import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sqlite3
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PhysiologicalReading:
    """Single physiological measurement"""
    participant_id: str
    timestamp: datetime
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    hrv_rmssd: Optional[float] = None
    hrv_pnn50: Optional[float] = None
    skin_conductance: Optional[float] = None
    skin_temperature: Optional[float] = None
    
    # Environmental context
    location: str = "indoor"  # indoor, outdoor
    weather: str = "sunny"    # sunny, cloudy, rainy
    temperature: float = 25.0  # Celsius
    humidity: float = 50.0     # percentage
    
    # Activity context
    activity: str = "resting"  # resting, walking, social, eating
    posture: str = "sitting"   # sitting, standing, lying
    
    # Self-reported emotion (ground truth)
    emotion: str = "neutral"   # calm, anxious, sad, happy, stressed, neutral
    emotion_intensity: int = 3  # 1-5 scale
    
    # Additional context
    notes: str = ""
    is_valid: bool = True


class PhysiologicalDataCollector:
    """Data collection system for physiological signals and emotions"""
    
    def __init__(self, db_path: str = "physiological_data.db"):
        self.db_path = db_path
        self.setup_database()
        
        # Data validation ranges
        self.validation_ranges = {
            'heart_rate': (40, 200),
            'systolic_bp': (70, 200), 
            'diastolic_bp': (40, 120),
            'hrv_rmssd': (0, 200),
            'hrv_pnn50': (0, 100),
            'skin_conductance': (0, 50),
            'skin_temperature': (20, 40)
        }
        
        # Emotion mappings
        self.emotion_classes = [
            'calm', 'anxious', 'sad', 'happy', 'stressed', 'neutral'
        ]
        
        self.activity_classes = [
            'resting', 'walking', 'social', 'eating', 'exercise'
        ]
        
        logger.info(f"Data collector initialized with database: {db_path}")
    
    def setup_database(self):
        """Initialize SQLite database for data storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create main data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS physiological_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                heart_rate REAL,
                systolic_bp REAL,
                diastolic_bp REAL,
                hrv_rmssd REAL,
                hrv_pnn50 REAL,
                skin_conductance REAL,
                skin_temperature REAL,
                location TEXT,
                weather TEXT,
                temperature REAL,
                humidity REAL,
                activity TEXT,
                posture TEXT,
                emotion TEXT,
                emotion_intensity INTEGER,
                notes TEXT,
                is_valid BOOLEAN,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create participants table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS participants (
                participant_id TEXT PRIMARY KEY,
                age INTEGER,
                gender TEXT,
                health_conditions TEXT,
                medications TEXT,
                baseline_hr REAL,
                baseline_bp_sys REAL,
                baseline_bp_dia REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create sessions table for trial organization
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_sessions (
                session_id TEXT PRIMARY KEY,
                participant_id TEXT,
                start_time TEXT,
                end_time TEXT,
                session_type TEXT,
                location TEXT,
                notes TEXT,
                FOREIGN KEY (participant_id) REFERENCES participants (participant_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("Database initialized successfully")
    
    def register_participant(self, participant_id: str, age: int, gender: str,
                           health_conditions: str = "", medications: str = "",
                           baseline_hr: float = 70, baseline_bp_sys: float = 120,
                           baseline_bp_dia: float = 80):
        """Register a new participant in the study"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO participants 
                (participant_id, age, gender, health_conditions, medications, 
                 baseline_hr, baseline_bp_sys, baseline_bp_dia)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (participant_id, age, gender, health_conditions, medications,
                  baseline_hr, baseline_bp_sys, baseline_bp_dia))
            
            conn.commit()
            logger.info(f"Registered participant: {participant_id}")
            
        except sqlite3.IntegrityError:
            logger.warning(f"Participant {participant_id} already exists")
        finally:
            conn.close()
    
    def validate_reading(self, reading: PhysiologicalReading) -> bool:
        """Validate physiological reading against normal ranges"""
        for field, (min_val, max_val) in self.validation_ranges.items():
            value = getattr(reading, field, None)
            if value is not None and not (min_val <= value <= max_val):
                logger.warning(f"Invalid {field}: {value} (expected {min_val}-{max_val})")
                return False
        return True
    
    def record_reading(self, reading: PhysiologicalReading) -> bool:
        """Record a single physiological reading"""
        # Validate reading
        is_valid = self.validate_reading(reading)
        reading.is_valid = is_valid
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO physiological_data 
                (participant_id, timestamp, heart_rate, systolic_bp, diastolic_bp,
                 hrv_rmssd, hrv_pnn50, skin_conductance, skin_temperature,
                 location, weather, temperature, humidity, activity, posture,
                 emotion, emotion_intensity, notes, is_valid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                reading.participant_id, reading.timestamp.isoformat(),
                reading.heart_rate, reading.systolic_bp, reading.diastolic_bp,
                reading.hrv_rmssd, reading.hrv_pnn50, reading.skin_conductance,
                reading.skin_temperature, reading.location, reading.weather,
                reading.temperature, reading.humidity, reading.activity,
                reading.posture, reading.emotion, reading.emotion_intensity,
                reading.notes, reading.is_valid
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error recording data: {e}")
            return False
        finally:
            conn.close()
    
    def get_participant_data(self, participant_id: str, 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve data for a specific participant"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM physiological_data 
            WHERE participant_id = ?
        '''
        params = [participant_id]
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date.isoformat())
        
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date.isoformat())
        
        query += ' ORDER BY timestamp'
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Retrieved {len(df)} records for participant {participant_id}")
        return df
    
    def export_for_training(self, output_path: str = "training_data.csv",
                           include_invalid: bool = False) -> pd.DataFrame:
        """Export data in format suitable for model training"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                participant_id, timestamp, heart_rate, systolic_bp, diastolic_bp,
                hrv_rmssd, hrv_pnn50, skin_conductance, skin_temperature,
                location, weather, temperature, humidity, activity, posture,
                emotion, emotion_intensity
            FROM physiological_data
        '''
        
        if not include_invalid:
            query += ' WHERE is_valid = 1'
        
        query += ' ORDER BY participant_id, timestamp'
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Export
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} records to {output_path}")
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw physiological data"""
        df = df.copy()
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # One-hot encode categorical variables
        df = pd.get_dummies(df, columns=['location', 'weather', 'activity'], prefix_sep='_')
        
        # Derived physiological features
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
        df['mean_arterial_pressure'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
        
        # Fill missing values with group means
        for col in ['hrv_rmssd', 'hrv_pnn50', 'skin_conductance', 'skin_temperature']:
            if col in df.columns:
                df[col] = df[col].fillna(df.groupby('participant_id')[col].transform('mean'))
        
        return df
    
    def generate_summary_report(self, participant_id: str) -> Dict:
        """Generate summary report for a participant"""
        df = self.get_participant_data(participant_id)
        
        if df.empty:
            return {"error": "No data found for participant"}
        
        # Basic statistics
        stats = {
            'total_readings': len(df),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat(),
                'days': (df['timestamp'].max() - df['timestamp'].min()).days
            },
            'physiological_summary': {
                'heart_rate': {
                    'mean': float(df['heart_rate'].mean()),
                    'std': float(df['heart_rate'].std()),
                    'min': float(df['heart_rate'].min()),
                    'max': float(df['heart_rate'].max())
                },
                'blood_pressure': {
                    'systolic_mean': float(df['systolic_bp'].mean()),
                    'diastolic_mean': float(df['diastolic_bp'].mean()),
                    'systolic_std': float(df['systolic_bp'].std()),
                    'diastolic_std': float(df['diastolic_bp'].std())
                }
            },
            'emotion_distribution': df['emotion'].value_counts().to_dict(),
            'activity_distribution': df['activity'].value_counts().to_dict(),
            'data_quality': {
                'valid_readings': int(df['is_valid'].sum()),
                'invalid_readings': int((~df['is_valid']).sum()),
                'completeness': float(df.notna().mean().mean())
            }
        }
        
        return stats
    
    def plot_participant_trends(self, participant_id: str, save_path: str = None):
        """Plot physiological trends for a participant"""
        df = self.get_participant_data(participant_id)
        
        if df.empty:
            logger.warning(f"No data found for participant {participant_id}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Physiological Trends - Participant {participant_id}', fontsize=16)
        
        # Heart rate over time
        axes[0, 0].plot(df['timestamp'], df['heart_rate'], alpha=0.7)
        axes[0, 0].set_title('Heart Rate Over Time')
        axes[0, 0].set_ylabel('Heart Rate (BPM)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Blood pressure over time
        axes[0, 1].plot(df['timestamp'], df['systolic_bp'], label='Systolic', alpha=0.7)
        axes[0, 1].plot(df['timestamp'], df['diastolic_bp'], label='Diastolic', alpha=0.7)
        axes[0, 1].set_title('Blood Pressure Over Time')
        axes[0, 1].set_ylabel('Blood Pressure (mmHg)')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Emotion distribution
        emotion_counts = df['emotion'].value_counts()
        axes[1, 0].pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Emotion Distribution')
        
        # Heart rate by emotion
        sns.boxplot(data=df, x='emotion', y='heart_rate', ax=axes[1, 1])
        axes[1, 1].set_title('Heart Rate by Emotion')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trends plot saved to {save_path}")
        
        plt.show()


class DataSimulator:
    """Simulate physiological data for testing purposes"""
    
    def __init__(self, collector: PhysiologicalDataCollector):
        self.collector = collector
    
    def simulate_participant_data(self, participant_id: str, days: int = 7,
                                readings_per_day: int = 48) -> None:
        """Simulate realistic physiological data for a participant"""
        
        # Register participant
        self.collector.register_participant(
            participant_id=participant_id,
            age=np.random.randint(65, 85),
            gender=np.random.choice(['male', 'female']),
            health_conditions="Mild hypertension",
            baseline_hr=70 + np.random.normal(0, 5),
            baseline_bp_sys=130 + np.random.normal(0, 10),
            baseline_bp_dia=85 + np.random.normal(0, 5)
        )
        
        start_date = datetime.now() - timedelta(days=days)
        
        # Simulate daily patterns
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Daily emotion pattern (morning = calm, afternoon = varied)
            daily_emotions = ['calm'] * 20 + ['neutral'] * 15 + \
                           ['anxious'] * 8 + ['happy'] * 3 + ['stressed'] * 2
            
            for reading_idx in range(readings_per_day):
                timestamp = current_date + timedelta(minutes=30 * reading_idx)
                hour = timestamp.hour
                
                # Circadian rhythm effects
                hr_base = 70 + 10 * np.sin(2 * np.pi * (hour - 6) / 24)
                bp_sys_base = 130 + 15 * np.sin(2 * np.pi * (hour - 6) / 24)
                bp_dia_base = 85 + 10 * np.sin(2 * np.pi * (hour - 6) / 24)
                
                # Emotion-dependent modulation
                emotion = np.random.choice(daily_emotions)
                
                if emotion == 'anxious':
                    hr_mod = np.random.normal(15, 5)
                    bp_mod = np.random.normal(10, 3)
                elif emotion == 'stressed':
                    hr_mod = np.random.normal(20, 5)
                    bp_mod = np.random.normal(15, 5)
                elif emotion == 'calm':
                    hr_mod = np.random.normal(-5, 2)
                    bp_mod = np.random.normal(-5, 2)
                else:
                    hr_mod = np.random.normal(0, 3)
                    bp_mod = np.random.normal(0, 2)
                
                # Activity effects
                if 6 <= hour <= 8 or 18 <= hour <= 20:
                    activity = 'walking'
                    hr_mod += np.random.normal(20, 5)
                elif 12 <= hour <= 14:
                    activity = 'eating' 
                    hr_mod += np.random.normal(5, 2)
                elif 15 <= hour <= 17:
                    activity = 'social'
                    hr_mod += np.random.normal(8, 3)
                else:
                    activity = 'resting'
                
                # Generate reading
                reading = PhysiologicalReading(
                    participant_id=participant_id,
                    timestamp=timestamp,
                    heart_rate=max(40, hr_base + hr_mod + np.random.normal(0, 2)),
                    systolic_bp=max(70, bp_sys_base + bp_mod + np.random.normal(0, 3)),
                    diastolic_bp=max(40, bp_dia_base + bp_mod + np.random.normal(0, 2)),
                    hrv_rmssd=max(10, 50 - hr_mod/2 + np.random.normal(0, 5)),
                    hrv_pnn50=max(0, 30 - hr_mod/3 + np.random.normal(0, 5)),
                    skin_conductance=2 + hr_mod/10 + np.random.normal(0, 0.5),
                    skin_temperature=36.5 + np.random.normal(0, 0.3),
                    location=np.random.choice(['indoor', 'outdoor'], p=[0.8, 0.2]),
                    weather=np.random.choice(['sunny', 'cloudy', 'rainy'], p=[0.5, 0.3, 0.2]),
                    temperature=25 + np.random.normal(0, 5),
                    humidity=50 + np.random.normal(0, 10),
                    activity=activity,
                    posture=np.random.choice(['sitting', 'standing'], p=[0.7, 0.3]),
                    emotion=emotion,
                    emotion_intensity=np.random.randint(2, 5)
                )
                
                self.collector.record_reading(reading)
        
        logger.info(f"Simulated {days * readings_per_day} readings for {participant_id}")


if __name__ == "__main__":
    # Example usage
    collector = PhysiologicalDataCollector()
    simulator = DataSimulator(collector)
    
    # Simulate data for 3 participants
    participants = ['CQ_001', 'CQ_002', 'CQ_003']
    
    for participant in participants:
        simulator.simulate_participant_data(participant, days=14, readings_per_day=48)
        
        # Generate report
        report = collector.generate_summary_report(participant)
        print(f"\nSummary for {participant}:")
        print(json.dumps(report, indent=2, default=str))
        
        # Plot trends
        collector.plot_participant_trends(participant, 
                                        save_path=f'trends_{participant}.png')
    
    # Export training data
    training_data = collector.export_for_training()
    print(f"\nTraining data exported: {training_data.shape}")
    print("Features:", list(training_data.columns))