import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from emotion_physiology_model import EmotionPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass 
class ScentAction:
    """Represents a scent release action"""
    scent_type: str  # 'lavender', 'citrus', 'mint', 'eucalyptus', 'none'
    intensity: float  # 0.0 to 1.0
    duration: int     # seconds


@dataclass
class PhysiologicalState:
    """Current physiological state of the user"""
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    hrv_rmssd: float
    skin_conductance: float
    skin_temperature: float
    
    # Derived metrics
    stress_index: float = 0.0
    arousal_level: float = 0.0
    
    # Trends (compared to last N readings)
    hr_trend: float = 0.0  # -1 to 1 (decreasing to increasing)
    bp_trend: float = 0.0
    stress_trend: float = 0.0


@dataclass
class EmotionalState:
    """Predicted emotional state"""
    predicted_emotion: str
    confidence: float
    probabilities: Dict[str, float]
    emotional_valence: float = 0.0  # -1 (negative) to 1 (positive)
    emotional_arousal: float = 0.0   # 0 (calm) to 1 (excited)


class ScentTherapyEnvironment(gym.Env):
    """
    Reinforcement Learning Environment for Scent-based Mood Optimization
    
    State Space:
    - Current physiological readings (6 values)
    - Physiological trends (3 values) 
    - Predicted emotional state (6 emotion probabilities)
    - Time context (4 values: hour_sin, hour_cos, day_sin, day_cos)
    - Recent scent history (5 values: last scent type and intensity)
    - User baseline comparison (3 values)
    Total: 27-dimensional continuous state space
    
    Action Space:
    - Scent type: 5 discrete choices (lavender, citrus, mint, eucalyptus, none)
    - Intensity: Continuous [0, 1]
    - Duration: Discrete [30, 60, 120, 300] seconds
    
    Reward Function:
    - Physiological improvement: +1 to +5 points
    - Emotional improvement: +1 to +3 points  
    - User feedback: +10 (positive) / -10 (negative)
    - Overuse penalty: -1 to -5 points
    - Safety considerations: -20 points for dangerous states
    """
    
    def __init__(self, participant_id: str, emotion_model_path: str = None,
                 max_steps: int = 100, real_time: bool = False):
        super().__init__()
        
        self.participant_id = participant_id
        self.max_steps = max_steps
        self.current_step = 0
        self.real_time = real_time
        
        # Load emotion prediction model
        self.emotion_predictor = EmotionPredictor()
        if emotion_model_path:
            self.emotion_predictor.load_model(emotion_model_path)
        
        # Action space: [scent_type, intensity, duration]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([4, 1, 3]),  # 5 scents, intensity 0-1, 4 duration options
            dtype=np.float32
        )
        
        # State space: 27-dimensional
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32
        )
        
        # Scent properties
        self.scent_types = {
            0: 'none',
            1: 'lavender',    # relaxing, sedative
            2: 'citrus',      # energizing, uplifting  
            3: 'mint',        # alerting, cooling
            4: 'eucalyptus'   # clearing, calming
        }
        
        self.scent_effects = {
            'lavender': {'stress_reduction': 0.8, 'arousal_change': -0.6},
            'citrus': {'stress_reduction': 0.3, 'arousal_change': 0.4},
            'mint': {'stress_reduction': 0.2, 'arousal_change': 0.8},
            'eucalyptus': {'stress_reduction': 0.6, 'arousal_change': -0.2},
            'none': {'stress_reduction': 0.0, 'arousal_change': 0.0}
        }
        
        # Duration options (seconds)
        self.duration_options = [30, 60, 120, 300]
        
        # User state tracking
        self.physiological_history = deque(maxlen=10)
        self.scent_history = deque(maxlen=5)
        self.user_feedback_history = deque(maxlen=20)
        
        # User baseline (learned over time)
        self.user_baseline = {
            'heart_rate': 75.0,
            'systolic_bp': 130.0,
            'diastolic_bp': 80.0,
            'hrv_rmssd': 25.0,
            'stress_threshold': 0.7
        }
        
        # Personalized scent preferences (learned)
        self.scent_preferences = {
            'lavender': 0.5,
            'citrus': 0.5, 
            'mint': 0.5,
            'eucalyptus': 0.5
        }
        
        # Safety constraints
        self.safety_limits = {
            'max_scent_per_hour': 6,
            'min_interval_seconds': 300,  # 5 minutes between scents
            'max_daily_scents': 30
        }
        
        # Current state
        self.current_physiological_state = None
        self.current_emotional_state = None
        self.last_scent_time = None
        self.daily_scent_count = 0
        self.hourly_scent_count = 0
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_actions = []
        self.total_reward = 0.0
        
        logger.info(f"Scent therapy environment initialized for participant {participant_id}")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.total_reward = 0.0
        self.episode_actions = []
        
        # Generate initial physiological state
        self.current_physiological_state = self._generate_initial_state()
        
        # Predict initial emotional state
        self.current_emotional_state = self._predict_emotion()
        
        # Reset tracking
        self.physiological_history.clear()
        self.scent_history.clear()
        self.physiological_history.append(self.current_physiological_state)
        
        # Reset safety counters
        self.daily_scent_count = 0
        self.hourly_scent_count = 0
        self.last_scent_time = None
        
        return self._get_state_vector()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
        self.current_step += 1
        
        # Parse action
        scent_action = self._parse_action(action)
        
        # Check safety constraints
        safety_violation = self._check_safety_constraints(scent_action)
        
        # Apply scent action if safe
        if not safety_violation:
            self._apply_scent_action(scent_action)
        
        # Simulate physiological response
        self._simulate_physiological_response(scent_action, safety_violation)
        
        # Update emotional state
        self.current_emotional_state = self._predict_emotion()
        
        # Calculate reward
        reward = self._calculate_reward(scent_action, safety_violation)
        self.total_reward += reward
        
        # Store action and state
        self.episode_actions.append(scent_action)
        self.physiological_history.append(self.current_physiological_state)
        
        # Check if episode is done
        done = (self.current_step >= self.max_steps) or self._check_termination()
        
        # Additional info
        info = {
            'scent_action': scent_action,
            'physiological_state': self.current_physiological_state,
            'emotional_state': self.current_emotional_state,
            'safety_violation': safety_violation,
            'episode_reward': self.total_reward,
            'stress_level': self._calculate_stress_index()
        }
        
        return self._get_state_vector(), reward, done, info
    
    def _parse_action(self, action: np.ndarray) -> ScentAction:
        """Parse continuous action vector to scent action"""
        scent_type_idx = int(np.clip(action[0], 0, 4))
        intensity = float(np.clip(action[1], 0, 1))
        duration_idx = int(np.clip(action[2], 0, 3))
        
        return ScentAction(
            scent_type=self.scent_types[scent_type_idx],
            intensity=intensity,
            duration=self.duration_options[duration_idx]
        )
    
    def _check_safety_constraints(self, scent_action: ScentAction) -> bool:
        """Check if action violates safety constraints"""
        current_time = datetime.now()
        
        # Check minimum interval
        if self.last_scent_time and scent_action.scent_type != 'none':
            time_since_last = (current_time - self.last_scent_time).total_seconds()
            if time_since_last < self.safety_limits['min_interval_seconds']:
                return True
        
        # Check hourly limit
        if scent_action.scent_type != 'none':
            if self.hourly_scent_count >= self.safety_limits['max_scent_per_hour']:
                return True
            
            # Check daily limit
            if self.daily_scent_count >= self.safety_limits['max_daily_scents']:
                return True
        
        # Check physiological safety
        if self._is_physiologically_unsafe():
            return True
        
        return False
    
    def _is_physiologically_unsafe(self) -> bool:
        """Check if current physiological state is unsafe"""
        if not self.current_physiological_state:
            return False
        
        # Check for extreme values
        if (self.current_physiological_state.heart_rate > 180 or 
            self.current_physiological_state.heart_rate < 50 or
            self.current_physiological_state.systolic_bp > 200 or
            self.current_physiological_state.systolic_bp < 90):
            return True
        
        return False
    
    def _apply_scent_action(self, scent_action: ScentAction):
        """Apply scent action and update tracking"""
        if scent_action.scent_type != 'none':
            self.scent_history.append(scent_action)
            self.last_scent_time = datetime.now()
            self.hourly_scent_count += 1
            self.daily_scent_count += 1
    
    def _simulate_physiological_response(self, scent_action: ScentAction, 
                                       safety_violation: bool):
        """Simulate physiological response to scent"""
        if not self.current_physiological_state:
            return
        
        # Base physiological drift (natural variation)
        hr_change = np.random.normal(0, 2)
        bp_sys_change = np.random.normal(0, 3)
        bp_dia_change = np.random.normal(0, 2)
        hrv_change = np.random.normal(0, 2)
        sc_change = np.random.normal(0, 0.1)
        temp_change = np.random.normal(0, 0.1)
        
        # Apply scent effects if no safety violation
        if not safety_violation and scent_action.scent_type != 'none':
            scent_effect = self.scent_effects[scent_action.scent_type]
            
            # Personalized response based on learned preferences
            effectiveness = self.scent_preferences.get(scent_action.scent_type, 0.5)
            
            # Calculate effects
            stress_reduction = (scent_effect['stress_reduction'] * 
                              scent_action.intensity * effectiveness)
            arousal_change = (scent_effect['arousal_change'] * 
                            scent_action.intensity * effectiveness)
            
            # Apply to physiological parameters
            hr_change -= stress_reduction * 10 + arousal_change * 15
            bp_sys_change -= stress_reduction * 8
            bp_dia_change -= stress_reduction * 5
            hrv_change += stress_reduction * 5
            sc_change -= stress_reduction * 0.5
        
        # Update physiological state
        self.current_physiological_state.heart_rate = max(40, 
            self.current_physiological_state.heart_rate + hr_change)
        
        self.current_physiological_state.systolic_bp = max(70,
            self.current_physiological_state.systolic_bp + bp_sys_change)
        
        self.current_physiological_state.diastolic_bp = max(40,
            self.current_physiological_state.diastolic_bp + bp_dia_change)
        
        self.current_physiological_state.hrv_rmssd = max(5,
            self.current_physiological_state.hrv_rmssd + hrv_change)
        
        self.current_physiological_state.skin_conductance = max(0,
            self.current_physiological_state.skin_conductance + sc_change)
        
        self.current_physiological_state.skin_temperature = np.clip(
            self.current_physiological_state.skin_temperature + temp_change, 32, 40)
        
        # Update derived metrics
        self._update_derived_metrics()
    
    def _update_derived_metrics(self):
        """Update derived physiological metrics"""
        if not self.current_physiological_state:
            return
        
        # Stress index (normalized combination of HR, BP, HRV, SC)
        hr_norm = (self.current_physiological_state.heart_rate - 60) / 40
        bp_norm = (self.current_physiological_state.systolic_bp - 120) / 40
        hrv_norm = -(self.current_physiological_state.hrv_rmssd - 30) / 30  # Lower HRV = higher stress
        sc_norm = (self.current_physiological_state.skin_conductance - 2) / 3
        
        self.current_physiological_state.stress_index = np.clip(
            np.mean([hr_norm, bp_norm, hrv_norm, sc_norm]), 0, 1)
        
        # Arousal level
        self.current_physiological_state.arousal_level = np.clip(
            (hr_norm + sc_norm) / 2, 0, 1)
        
        # Calculate trends if enough history
        if len(self.physiological_history) >= 3:
            recent_states = list(self.physiological_history)[-3:]
            
            # HR trend
            hr_values = [s.heart_rate for s in recent_states]
            self.current_physiological_state.hr_trend = np.polyfit(
                range(len(hr_values)), hr_values, 1)[0] / 10  # normalize
            
            # BP trend  
            bp_values = [s.systolic_bp for s in recent_states]
            self.current_physiological_state.bp_trend = np.polyfit(
                range(len(bp_values)), bp_values, 1)[0] / 10
            
            # Stress trend
            stress_values = [s.stress_index for s in recent_states]
            self.current_physiological_state.stress_trend = np.polyfit(
                range(len(stress_values)), stress_values, 1)[0]
    
    def _predict_emotion(self) -> EmotionalState:
        """Predict emotional state using the trained model"""
        if not self.current_physiological_state:
            return EmotionalState(
                predicted_emotion='neutral',
                confidence=0.5,
                probabilities={'neutral': 1.0}
            )
        
        # Prepare input data for emotion model
        physiological_data = np.array([[
            self.current_physiological_state.heart_rate,
            self.current_physiological_state.systolic_bp,
            self.current_physiological_state.diastolic_bp,
            self.current_physiological_state.hrv_rmssd,
            self.current_physiological_state.skin_conductance,
            self.current_physiological_state.skin_temperature
        ] + [0] * 12])  # Pad with zeros for other features
        
        # Make prediction
        try:
            prediction = self.emotion_predictor.predict_emotion(physiological_data)
            
            # Calculate emotional valence and arousal
            valence_mapping = {
                'happy': 0.8, 'calm': 0.6, 'neutral': 0.0,
                'anxious': -0.4, 'stressed': -0.6, 'sad': -0.8
            }
            
            arousal_mapping = {
                'stressed': 0.9, 'anxious': 0.8, 'happy': 0.6,
                'neutral': 0.3, 'sad': 0.2, 'calm': 0.1
            }
            
            emotion = prediction['predicted_emotion']
            
            return EmotionalState(
                predicted_emotion=emotion,
                confidence=prediction['confidence'],
                probabilities=prediction['probabilities'],
                emotional_valence=valence_mapping.get(emotion, 0.0),
                emotional_arousal=arousal_mapping.get(emotion, 0.3)
            )
            
        except Exception as e:
            logger.warning(f"Emotion prediction failed: {e}")
            return EmotionalState(
                predicted_emotion='neutral',
                confidence=0.5,
                probabilities={'neutral': 1.0}
            )
    
    def _calculate_reward(self, scent_action: ScentAction, 
                         safety_violation: bool) -> float:
        """Calculate reward based on physiological and emotional improvements"""
        reward = 0.0
        
        # Safety penalty
        if safety_violation:
            reward -= 20
            return reward
        
        # Physiological improvement reward
        if len(self.physiological_history) >= 2:
            prev_state = self.physiological_history[-2]
            current_state = self.current_physiological_state
            
            # Heart rate improvement (towards baseline)
            hr_baseline = self.user_baseline['heart_rate']
            prev_hr_diff = abs(prev_state.heart_rate - hr_baseline)
            current_hr_diff = abs(current_state.heart_rate - hr_baseline)
            hr_improvement = prev_hr_diff - current_hr_diff
            reward += hr_improvement / 5  # Scale factor
            
            # Blood pressure improvement
            bp_baseline = self.user_baseline['systolic_bp']
            prev_bp_diff = abs(prev_state.systolic_bp - bp_baseline)
            current_bp_diff = abs(current_state.systolic_bp - bp_baseline)
            bp_improvement = prev_bp_diff - current_bp_diff
            reward += bp_improvement / 10
            
            # HRV improvement (higher is better)
            hrv_improvement = current_state.hrv_rmssd - prev_state.hrv_rmssd
            reward += hrv_improvement / 5
            
            # Stress reduction
            stress_reduction = prev_state.stress_index - current_state.stress_index
            reward += stress_reduction * 10  # High weight for stress reduction
        
        # Emotional improvement reward
        if self.current_emotional_state:
            # Reward positive emotions
            positive_emotions = ['happy', 'calm']
            if self.current_emotional_state.predicted_emotion in positive_emotions:
                reward += self.current_emotional_state.confidence * 3
            
            # Penalty for negative emotions
            negative_emotions = ['stressed', 'anxious', 'sad']
            if self.current_emotional_state.predicted_emotion in negative_emotions:
                reward -= self.current_emotional_state.confidence * 2
        
        # Scent usage efficiency
        if scent_action.scent_type != 'none':
            # Small penalty for using scents (encourage efficiency)
            reward -= 0.5
            
            # Bonus for effective scent usage
            scent_preference = self.scent_preferences.get(scent_action.scent_type, 0.5)
            reward += scent_preference * scent_action.intensity * 2
        
        # Overuse penalty
        if self.hourly_scent_count > 4:
            reward -= (self.hourly_scent_count - 4) * 2
        
        # Consistency reward (small bonus for maintaining good state)
        if self.current_physiological_state.stress_index < 0.3:
            reward += 1
        
        return np.clip(reward, -50, 50)  # Clip extreme rewards
    
    def _calculate_stress_index(self) -> float:
        """Calculate current stress index"""
        if self.current_physiological_state:
            return self.current_physiological_state.stress_index
        return 0.5
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate early"""
        # Terminate if user reaches very low stress for extended period
        if (len(self.physiological_history) >= 5 and
            all(s.stress_index < 0.2 for s in list(self.physiological_history)[-5:])):
            return True
        
        # Terminate if unsafe physiological state persists
        if (len(self.physiological_history) >= 3 and
            all(s.heart_rate > 180 or s.systolic_bp > 200 
                for s in list(self.physiological_history)[-3:])):
            return True
        
        return False
    
    def _get_state_vector(self) -> np.ndarray:
        """Convert current state to observation vector"""
        if not self.current_physiological_state or not self.current_emotional_state:
            return np.zeros(27, dtype=np.float32)
        
        state_vector = []
        
        # Physiological readings (6 values)
        state_vector.extend([
            self.current_physiological_state.heart_rate / 200,  # normalize
            self.current_physiological_state.systolic_bp / 200,
            self.current_physiological_state.diastolic_bp / 120,
            self.current_physiological_state.hrv_rmssd / 100,
            self.current_physiological_state.skin_conductance / 10,
            self.current_physiological_state.skin_temperature / 40
        ])
        
        # Physiological trends (3 values)
        state_vector.extend([
            np.clip(self.current_physiological_state.hr_trend, -1, 1),
            np.clip(self.current_physiological_state.bp_trend, -1, 1),
            np.clip(self.current_physiological_state.stress_trend, -1, 1)
        ])
        
        # Emotion probabilities (6 values)
        emotion_probs = [
            self.current_emotional_state.probabilities.get(emotion, 0.0)
            for emotion in ['calm', 'anxious', 'sad', 'happy', 'stressed', 'neutral']
        ]
        state_vector.extend(emotion_probs)
        
        # Time context (4 values)
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        state_vector.extend([
            np.sin(2 * np.pi * current_hour / 24),
            np.cos(2 * np.pi * current_hour / 24),
            np.sin(2 * np.pi * current_day / 7),
            np.cos(2 * np.pi * current_day / 7)
        ])
        
        # Recent scent history (5 values: type, intensity, time since, count, effectiveness)
        if self.scent_history:
            last_scent = self.scent_history[-1]
            scent_type_encoded = list(self.scent_types.values()).index(last_scent.scent_type)
            time_since = (datetime.now() - self.last_scent_time).total_seconds() / 3600  # hours
            effectiveness = self.scent_preferences.get(last_scent.scent_type, 0.5)
            
            state_vector.extend([
                scent_type_encoded / 4,  # normalize
                last_scent.intensity,
                min(time_since, 24) / 24,  # normalize to 24 hours
                min(len(self.scent_history), 5) / 5,  # normalize
                effectiveness
            ])
        else:
            state_vector.extend([0, 0, 1, 0, 0.5])  # default values
        
        # User baseline comparison (3 values)
        hr_baseline_diff = (self.current_physiological_state.heart_rate - 
                           self.user_baseline['heart_rate']) / 50
        bp_baseline_diff = (self.current_physiological_state.systolic_bp -
                           self.user_baseline['systolic_bp']) / 50
        stress_level = self.current_physiological_state.stress_index
        
        state_vector.extend([
            np.clip(hr_baseline_diff, -2, 2),
            np.clip(bp_baseline_diff, -2, 2), 
            stress_level
        ])
        
        return np.array(state_vector, dtype=np.float32)
    
    def _generate_initial_state(self) -> PhysiologicalState:
        """Generate realistic initial physiological state"""
        return PhysiologicalState(
            heart_rate=self.user_baseline['heart_rate'] + np.random.normal(0, 8),
            systolic_bp=self.user_baseline['systolic_bp'] + np.random.normal(0, 10),
            diastolic_bp=self.user_baseline['diastolic_bp'] + np.random.normal(0, 5),
            hrv_rmssd=25 + np.random.normal(0, 5),
            skin_conductance=2 + np.random.normal(0, 0.5),
            skin_temperature=36.5 + np.random.normal(0, 0.3)
        )
    
    def get_user_feedback(self, feedback: str) -> float:
        """Process user feedback and return reward adjustment"""
        feedback_reward = 0.0
        
        if feedback.lower() in ['good', 'better', 'relaxed', 'calm']:
            feedback_reward = 10
            # Update scent preferences positively for recent scents
            if self.scent_history:
                recent_scent = self.scent_history[-1].scent_type
                if recent_scent != 'none':
                    self.scent_preferences[recent_scent] = min(1.0,
                        self.scent_preferences[recent_scent] + 0.1)
                    
        elif feedback.lower() in ['bad', 'worse', 'uncomfortable', 'anxious']:
            feedback_reward = -10
            # Update scent preferences negatively
            if self.scent_history:
                recent_scent = self.scent_history[-1].scent_type
                if recent_scent != 'none':
                    self.scent_preferences[recent_scent] = max(0.0,
                        self.scent_preferences[recent_scent] - 0.1)
        
        self.user_feedback_history.append((feedback, feedback_reward))
        return feedback_reward
    
    def update_user_baseline(self, new_readings: Dict[str, float]):
        """Update user baseline with new physiological data"""
        alpha = 0.1  # Learning rate
        
        for key, new_value in new_readings.items():
            if key in self.user_baseline:
                self.user_baseline[key] = (alpha * new_value + 
                                         (1 - alpha) * self.user_baseline[key])
    
    def get_episode_summary(self) -> Dict:
        """Get summary of current episode"""
        return {
            'total_reward': self.total_reward,
            'total_steps': self.current_step,
            'scents_used': len(self.episode_actions),
            'average_stress': np.mean([s.stress_index for s in self.physiological_history]),
            'scent_distribution': {scent: sum(1 for a in self.episode_actions 
                                            if a.scent_type == scent) 
                                 for scent in self.scent_types.values()},
            'user_preferences': self.scent_preferences.copy()
        }


if __name__ == "__main__":
    # Example usage
    env = ScentTherapyEnvironment(participant_id="CQ_001")
    
    # Test environment
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a few steps
    for step in range(5):
        # Random action
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"Action: {info['scent_action']}")
        print(f"Reward: {reward:.2f}")
        print(f"Stress level: {info['stress_level']:.2f}")
        print(f"Emotion: {info['emotional_state'].predicted_emotion}")
        
        if done:
            break
    
    # Get episode summary
    summary = env.get_episode_summary()
    print(f"\nEpisode Summary:")
    print(json.dumps(summary, indent=2, default=str))