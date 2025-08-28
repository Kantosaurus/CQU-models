"""
Tolerance Building Recommendation Engine
Group 2 - Chongqing Food Spice Prediction

This module provides intelligent recommendations for building spice tolerance:
1. Progressive spice level recommendations
2. Tolerance building pathways
3. Safety guidelines and pacing recommendations
4. Dish selection for tolerance training
5. Progress tracking and encouragement
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import sqlite3
from dataclasses import dataclass
from collections import defaultdict
import logging

from data_collection import SpiceDataCollector, DishData
from user_profiling import UserProfilingSystem, ToleranceProfile

logger = logging.getLogger(__name__)

@dataclass
class ToleranceBuildingPlan:
    """Personalized tolerance building plan"""
    user_id: str
    current_tolerance: int
    target_tolerance: int
    estimated_timeline_weeks: int
    weekly_goals: List[Dict[str, any]]
    recommended_dishes: List[Tuple[str, int, str]]  # (dish_id, week, reason)
    safety_guidelines: List[str]
    progress_milestones: List[Dict[str, any]]
    created_at: datetime

@dataclass
class ProgressUpdate:
    """User's progress in tolerance building"""
    user_id: str
    current_week: int
    tolerance_gain: float
    dishes_tried: List[str]
    comfort_level: int  # 1-5 scale
    side_effects: List[str]
    confidence_change: float
    timestamp: datetime

class ToleranceBuildingEngine:
    """Engine for creating personalized spice tolerance building programs"""
    
    def __init__(self, db_path: str = "data/spice_database.db"):
        self.db_path = db_path
        self.data_collector = SpiceDataCollector(db_path)
        self.user_profiler = UserProfilingSystem(db_path)
        self.setup_tolerance_tables()
        
        # Tolerance building parameters
        self.max_tolerance_gain_per_week = 0.3  # Maximum safe weekly increase
        self.min_comfort_threshold = 3  # Minimum comfort level to proceed
        self.default_program_length_weeks = 12
        
        # Safety guidelines
        self.safety_rules = {
            'max_weekly_increase': 0.5,
            'rest_days_between_sessions': 2,
            'hydration_reminder': True,
            'dairy_recommendation': True,
            'gradual_progression_only': True
        }
        
        # Chongqing spice progression pathway
        self.spice_progression_path = {
            1: {  # Very Low -> Low
                'target_ingredients': ['mild doubanjiang', 'green peppers'],
                'recommended_dishes': ['sweet_sour_fish', 'mild_mapo_tofu'],
                'duration_weeks': 3
            },
            2: {  # Low -> Medium-Low  
                'target_ingredients': ['er_jin_tiao', 'mild_hua_jiao'],
                'recommended_dishes': ['kung_pao_chicken', 'mild_dan_dan_noodles'],
                'duration_weeks': 3
            },
            3: {  # Medium-Low -> Medium
                'target_ingredients': ['tian_jiao', 'standard_doubanjiang'],
                'recommended_dishes': ['mapo_tofu', 'spicy_fish_hotpot'],
                'duration_weeks': 3
            },
            4: {  # Medium -> High
                'target_ingredients': ['xiao_mi_la', 'strong_hua_jiao'],
                'recommended_dishes': ['chongqing_hotpot', 'laziji'],
                'duration_weeks': 4
            },
            5: {  # High -> Very High
                'target_ingredients': ['chao_tian_jiao', 'ghost_peppers'],
                'recommended_dishes': ['extreme_hotpot', 'super_spicy_dry_pot'],
                'duration_weeks': 6
            }
        }
    
    def setup_tolerance_tables(self):
        """Setup database tables for tolerance building tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tolerance_plans (
            plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            current_tolerance INTEGER,
            target_tolerance INTEGER,
            estimated_timeline_weeks INTEGER,
            weekly_goals TEXT,  -- JSON
            recommended_dishes TEXT,  -- JSON
            safety_guidelines TEXT,  -- JSON
            progress_milestones TEXT,  -- JSON
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tolerance_progress (
            progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            plan_id INTEGER,
            current_week INTEGER,
            tolerance_gain REAL,
            dishes_tried TEXT,  -- JSON
            comfort_level INTEGER,
            side_effects TEXT,  -- JSON
            confidence_change REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (plan_id) REFERENCES tolerance_plans (plan_id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tolerance_achievements (
            achievement_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            achievement_type TEXT,
            achievement_description TEXT,
            spice_level_reached INTEGER,
            dish_conquered TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_tolerance_building_plan(self, user_id: str, target_tolerance: int = None) -> ToleranceBuildingPlan:
        """Create personalized tolerance building plan"""
        logger.info(f"Creating tolerance building plan for user {user_id}")
        
        # Get current user profile
        user_profile = self.user_profiler.get_user_profile(user_id)
        if not user_profile:
            user_profile = self.user_profiler.build_complete_profile(user_id)
        
        current_tolerance = user_profile.tolerance_level
        
        # Determine target tolerance if not specified
        if target_tolerance is None:
            target_tolerance = min(5, current_tolerance + 1)  # Increase by 1 level by default
        
        # Validate target
        if target_tolerance <= current_tolerance:
            raise ValueError(f"Target tolerance ({target_tolerance}) must be higher than current ({current_tolerance})")
        
        if target_tolerance > 5:
            target_tolerance = 5
        
        # Calculate timeline
        tolerance_difference = target_tolerance - current_tolerance
        estimated_weeks = max(4, int(tolerance_difference * 4))  # At least 4 weeks, 4 weeks per level
        
        # Generate weekly goals
        weekly_goals = self._generate_weekly_goals(current_tolerance, target_tolerance, estimated_weeks)
        
        # Select recommended dishes
        recommended_dishes = self._select_progressive_dishes(current_tolerance, target_tolerance, estimated_weeks)
        
        # Generate safety guidelines
        safety_guidelines = self._generate_safety_guidelines(current_tolerance, target_tolerance)
        
        # Create progress milestones
        progress_milestones = self._create_progress_milestones(current_tolerance, target_tolerance, estimated_weeks)
        
        plan = ToleranceBuildingPlan(
            user_id=user_id,
            current_tolerance=current_tolerance,
            target_tolerance=target_tolerance,
            estimated_timeline_weeks=estimated_weeks,
            weekly_goals=weekly_goals,
            recommended_dishes=recommended_dishes,
            safety_guidelines=safety_guidelines,
            progress_milestones=progress_milestones,
            created_at=datetime.now()
        )
        
        # Store plan
        self._store_tolerance_plan(plan)
        
        return plan
    
    def _generate_weekly_goals(self, current: int, target: int, weeks: int) -> List[Dict[str, any]]:
        """Generate weekly progression goals"""
        weekly_goals = []
        tolerance_increment = (target - current) / weeks
        
        for week in range(1, weeks + 1):
            week_target_tolerance = current + (tolerance_increment * week)
            
            # Determine focus for this week
            if week <= weeks * 0.3:
                focus = "foundation_building"
                goal_type = "Try milder dishes to build confidence"
            elif week <= weeks * 0.7:
                focus = "steady_progression"
                goal_type = "Gradually increase spice levels"
            else:
                focus = "target_achievement"
                goal_type = "Practice target-level dishes"
            
            weekly_goal = {
                'week': week,
                'target_tolerance': round(week_target_tolerance, 1),
                'focus': focus,
                'goal_description': goal_type,
                'dishes_to_try': 2 if week <= weeks * 0.5 else 3,
                'rest_days': 2,
                'comfort_check': True
            }
            
            # Add specific milestones
            if week % 3 == 0:  # Every 3 weeks
                weekly_goal['milestone_check'] = True
                weekly_goal['reassessment'] = True
            
            weekly_goals.append(weekly_goal)
        
        return weekly_goals
    
    def _select_progressive_dishes(self, current: int, target: int, weeks: int) -> List[Tuple[str, int, str]]:
        """Select dishes for progressive tolerance building"""
        recommended_dishes = []
        
        # Get available dishes from database
        available_dishes = self._get_dishes_by_spice_range(current, target + 1)
        
        if not available_dishes:
            logger.warning("No dishes available for tolerance building")
            return []
        
        tolerance_increment = (target - current) / weeks
        
        for week in range(1, weeks + 1):
            week_target_tolerance = current + (tolerance_increment * week)
            
            # Select dishes appropriate for this week's tolerance level
            week_spice_level = max(current, min(target, int(week_target_tolerance)))
            
            # Find dishes at or slightly above current comfort level
            suitable_dishes = [
                dish for dish in available_dishes 
                if dish.base_spice_level == week_spice_level or 
                   dish.base_spice_level == week_spice_level + 1
            ]
            
            if suitable_dishes:
                # Select 1-2 dishes for this week
                num_dishes = 1 if week <= weeks * 0.3 else 2
                selected = np.random.choice(suitable_dishes, min(num_dishes, len(suitable_dishes)), replace=False)
                
                for dish in selected:
                    reason = self._get_recommendation_reason(dish, week, week_target_tolerance)
                    recommended_dishes.append((dish.dish_id, week, reason))
        
        return recommended_dishes
    
    def _get_dishes_by_spice_range(self, min_spice: int, max_spice: int) -> List[DishData]:
        """Get dishes within spice level range"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT * FROM dishes 
        WHERE base_spice_level BETWEEN ? AND ?
        ORDER BY base_spice_level, name
        '''
        
        df = pd.read_sql_query(query, conn, params=(min_spice, max_spice))
        conn.close()
        
        dishes = []
        for _, row in df.iterrows():
            dish = DishData(
                dish_id=row['dish_id'],
                name=row['name'],
                restaurant_id=row['restaurant_id'],
                cuisine_type=row['cuisine_type'],
                main_ingredients=eval(row['main_ingredients']) if row['main_ingredients'] else [],
                cooking_method=row['cooking_method'],
                spice_ingredients=eval(row['spice_ingredients']) if row['spice_ingredients'] else [],
                lab_scoville=row['lab_scoville'],
                base_spice_level=row['base_spice_level'],
                price=row['price'],
                description=row['description']
            )
            dishes.append(dish)
        
        return dishes
    
    def _get_recommendation_reason(self, dish: DishData, week: int, target_tolerance: float) -> str:
        """Generate reason for dish recommendation"""
        reasons = []
        
        if week <= 3:
            reasons.append("Foundation building")
        elif week <= 6:
            reasons.append("Steady progression")
        else:
            reasons.append("Target level practice")
        
        # Add specific reasons based on dish characteristics
        if dish.cooking_method == 'steam':
            reasons.append("milder cooking method")
        elif dish.cooking_method == 'hot_pot':
            reasons.append("customizable spice level")
        
        if 'doubanjiang' in dish.spice_ingredients:
            reasons.append("classic Sichuan flavor base")
        
        if dish.price and dish.price < 25:
            reasons.append("budget-friendly option")
        
        return "; ".join(reasons)
    
    def _generate_safety_guidelines(self, current: int, target: int) -> List[str]:
        """Generate safety guidelines for tolerance building"""
        guidelines = [
            "🥛 Keep dairy products (milk, yogurt) handy to neutralize excessive spice",
            "💧 Stay well-hydrated before, during, and after spicy meals",
            "⏱️ Allow 2-3 days between intense spicy meals for recovery",
            "🛑 Stop immediately if you experience severe stomach pain or nausea",
            "📈 Progress gradually - rushing can cause setbacks",
            "🍚 Always eat spicy food with rice or bread to buffer the heat",
            "🌙 Avoid very spicy food late at night to prevent sleep disruption"
        ]
        
        # Add specific guidelines based on tolerance levels
        if current == 1:
            guidelines.append("🥄 Start with small portions to test your reaction")
            guidelines.append("👥 Try spicy foods with friends for support and safety")
        
        if target >= 4:
            guidelines.append("🏥 Be aware of your limits - high spice levels can cause physical stress")
            guidelines.append("⚕️ Consider consulting a doctor if you have digestive issues")
        
        if target == 5:
            guidelines.append("🌶️ Extreme spice levels are for experienced eaters only")
            guidelines.append("📞 Have someone available to help if you have adverse reactions")
        
        return guidelines
    
    def _create_progress_milestones(self, current: int, target: int, weeks: int) -> List[Dict[str, any]]:
        """Create progress milestones for motivation"""
        milestones = []
        
        # Week milestones
        milestone_weeks = [weeks // 4, weeks // 2, 3 * weeks // 4, weeks]
        milestone_names = ["First Quarter", "Halfway Point", "Final Stretch", "Goal Achievement"]
        
        tolerance_increment = (target - current) / weeks
        
        for i, week in enumerate(milestone_weeks):
            milestone_tolerance = current + (tolerance_increment * week)
            
            milestone = {
                'week': week,
                'name': milestone_names[i],
                'target_tolerance': round(milestone_tolerance, 1),
                'celebration': self._get_celebration_suggestion(i, milestone_tolerance),
                'reward_suggestion': self._get_reward_suggestion(i),
                'assessment': "Complete tolerance reassessment"
            }
            milestones.append(milestone)
        
        return milestones
    
    def _get_celebration_suggestion(self, milestone_index: int, tolerance_level: float) -> str:
        """Get celebration suggestion for milestones"""
        celebrations = [
            "🎉 You've started your spice journey! Try a celebratory mild Sichuan dish.",
            "🔥 Halfway there! Challenge yourself with a medium-spicy hot pot.",
            "💪 Almost at your goal! You're becoming a spice warrior.",
            "🏆 Congratulations! You've reached your target tolerance level!"
        ]
        return celebrations[milestone_index]
    
    def _get_reward_suggestion(self, milestone_index: int) -> str:
        """Get reward suggestion for reaching milestones"""
        rewards = [
            "Treat yourself to a high-quality Sichuan peppercorn set",
            "Visit a famous Chongqing restaurant to test your progress",
            "Share your achievement on social media with #SpiceJourney",
            "Plan a Chongqing food tour to celebrate your success"
        ]
        return rewards[milestone_index]
    
    def update_progress(self, user_id: str, week: int, comfort_level: int, 
                       dishes_tried: List[str], side_effects: List[str] = None) -> ProgressUpdate:
        """Update user's tolerance building progress"""
        # Get current plan
        plan = self.get_active_plan(user_id)
        if not plan:
            raise ValueError(f"No active tolerance building plan for user {user_id}")
        
        # Calculate tolerance gain
        expected_tolerance = plan.current_tolerance + ((plan.target_tolerance - plan.current_tolerance) * week / plan.estimated_timeline_weeks)
        
        # Estimate actual tolerance based on comfort level and dishes tried
        actual_tolerance = self._estimate_current_tolerance(user_id, comfort_level, dishes_tried)
        tolerance_gain = actual_tolerance - plan.current_tolerance
        
        # Calculate confidence change
        confidence_change = self._calculate_confidence_change(comfort_level, len(dishes_tried))
        
        progress = ProgressUpdate(
            user_id=user_id,
            current_week=week,
            tolerance_gain=tolerance_gain,
            dishes_tried=dishes_tried,
            comfort_level=comfort_level,
            side_effects=side_effects or [],
            confidence_change=confidence_change,
            timestamp=datetime.now()
        )
        
        # Store progress
        self._store_progress_update(progress, plan)
        
        # Check for achievements
        self._check_achievements(user_id, progress, plan)
        
        # Provide recommendations for next week
        next_week_recommendations = self._generate_next_week_recommendations(progress, plan)
        
        return progress
    
    def _estimate_current_tolerance(self, user_id: str, comfort_level: int, dishes_tried: List[str]) -> float:
        """Estimate user's current tolerance based on recent performance"""
        if not dishes_tried:
            return self.user_profiler.get_user_profile(user_id).tolerance_level
        
        # Get spice levels of dishes tried
        conn = sqlite3.connect(self.db_path)
        
        dish_placeholders = ','.join(['?' for _ in dishes_tried])
        query = f'SELECT AVG(base_spice_level) FROM dishes WHERE dish_id IN ({dish_placeholders})'
        
        result = pd.read_sql_query(query, conn, params=dishes_tried)
        conn.close()
        
        avg_dish_spice = result.iloc[0, 0] if not result.empty and not pd.isna(result.iloc[0, 0]) else 3.0
        
        # Adjust based on comfort level
        comfort_adjustment = (comfort_level - 3) * 0.3  # -0.6 to +0.6 adjustment
        estimated_tolerance = avg_dish_spice + comfort_adjustment
        
        return max(1, min(5, estimated_tolerance))
    
    def _calculate_confidence_change(self, comfort_level: int, dishes_count: int) -> float:
        """Calculate change in user's confidence"""
        base_confidence_change = (comfort_level - 3) * 0.1  # More comfort = more confidence
        experience_bonus = min(0.1, dishes_count * 0.02)  # Bonus for trying more dishes
        
        return base_confidence_change + experience_bonus
    
    def _generate_next_week_recommendations(self, progress: ProgressUpdate, plan: ToleranceBuildingPlan) -> Dict[str, any]:
        """Generate recommendations for the next week"""
        recommendations = {
            'continue_current_pace': True,
            'suggested_dishes': [],
            'focus_areas': [],
            'warnings': []
        }
        
        # Analyze progress
        if progress.comfort_level < 3:
            recommendations['continue_current_pace'] = False
            recommendations['warnings'].append("Consider slowing down the progression")
            recommendations['focus_areas'].append("Build confidence with current spice level")
        elif progress.comfort_level >= 4:
            recommendations['focus_areas'].append("Ready to increase spice level")
        
        if progress.side_effects:
            recommendations['warnings'].append("Monitor side effects carefully")
            recommendations['continue_current_pace'] = False
        
        return recommendations
    
    def _check_achievements(self, user_id: str, progress: ProgressUpdate, plan: ToleranceBuildingPlan):
        """Check and record achievements"""
        achievements = []
        
        # Check for milestone achievements
        weeks_completed = progress.current_week
        total_weeks = plan.estimated_timeline_weeks
        
        if weeks_completed >= total_weeks // 4 and weeks_completed < total_weeks // 2:
            achievements.append("First Quarter Complete")
        elif weeks_completed >= total_weeks // 2 and weeks_completed < 3 * total_weeks // 4:
            achievements.append("Halfway Point Reached")
        elif weeks_completed >= 3 * total_weeks // 4 and weeks_completed < total_weeks:
            achievements.append("Final Stretch")
        elif weeks_completed >= total_weeks:
            achievements.append("Goal Achieved")
        
        # Check for comfort level achievements
        if progress.comfort_level >= 4:
            achievements.append("High Comfort Level")
        
        # Check for dish achievements
        if len(progress.dishes_tried) >= 5:
            achievements.append("Adventurous Eater")
        
        # Store achievements
        for achievement in achievements:
            self._store_achievement(user_id, achievement, progress.tolerance_gain)
    
    def get_active_plan(self, user_id: str) -> Optional[ToleranceBuildingPlan]:
        """Get user's active tolerance building plan"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT * FROM tolerance_plans 
        WHERE user_id = ? AND is_active = 1
        ORDER BY created_at DESC
        LIMIT 1
        '''
        
        result = pd.read_sql_query(query, conn, params=(user_id,))
        conn.close()
        
        if result.empty:
            return None
        
        row = result.iloc[0]
        
        return ToleranceBuildingPlan(
            user_id=row['user_id'],
            current_tolerance=row['current_tolerance'],
            target_tolerance=row['target_tolerance'],
            estimated_timeline_weeks=row['estimated_timeline_weeks'],
            weekly_goals=eval(row['weekly_goals']) if row['weekly_goals'] else [],
            recommended_dishes=eval(row['recommended_dishes']) if row['recommended_dishes'] else [],
            safety_guidelines=eval(row['safety_guidelines']) if row['safety_guidelines'] else [],
            progress_milestones=eval(row['progress_milestones']) if row['progress_milestones'] else [],
            created_at=datetime.fromisoformat(row['created_at'])
        )
    
    def _store_tolerance_plan(self, plan: ToleranceBuildingPlan):
        """Store tolerance building plan in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO tolerance_plans
        (user_id, current_tolerance, target_tolerance, estimated_timeline_weeks,
         weekly_goals, recommended_dishes, safety_guidelines, progress_milestones)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            plan.user_id, plan.current_tolerance, plan.target_tolerance,
            plan.estimated_timeline_weeks, str(plan.weekly_goals),
            str(plan.recommended_dishes), str(plan.safety_guidelines),
            str(plan.progress_milestones)
        ))
        
        conn.commit()
        conn.close()
    
    def _store_progress_update(self, progress: ProgressUpdate, plan: ToleranceBuildingPlan):
        """Store progress update in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get plan ID
        cursor.execute('SELECT plan_id FROM tolerance_plans WHERE user_id = ? AND is_active = 1', (progress.user_id,))
        plan_id = cursor.fetchone()[0]
        
        cursor.execute('''
        INSERT INTO tolerance_progress
        (user_id, plan_id, current_week, tolerance_gain, dishes_tried,
         comfort_level, side_effects, confidence_change)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            progress.user_id, plan_id, progress.current_week,
            progress.tolerance_gain, str(progress.dishes_tried),
            progress.comfort_level, str(progress.side_effects),
            progress.confidence_change
        ))
        
        conn.commit()
        conn.close()
    
    def _store_achievement(self, user_id: str, achievement: str, tolerance_level: float):
        """Store user achievement"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO tolerance_achievements
        (user_id, achievement_type, achievement_description, spice_level_reached)
        VALUES (?, ?, ?, ?)
        ''', (user_id, achievement, achievement, tolerance_level))
        
        conn.commit()
        conn.close()

if __name__ == "__main__":
    # Example usage
    tolerance_engine = ToleranceBuildingEngine()
    
    # Create tolerance building plan
    plan = tolerance_engine.create_tolerance_building_plan("U001", target_tolerance=4)
    print(f"Created tolerance building plan:")
    print(f"  Current tolerance: {plan.current_tolerance}")
    print(f"  Target tolerance: {plan.target_tolerance}")
    print(f"  Timeline: {plan.estimated_timeline_weeks} weeks")
    print(f"  Safety guidelines: {len(plan.safety_guidelines)} guidelines")
    print(f"  Recommended dishes: {len(plan.recommended_dishes)} dishes")
    
    # Simulate progress update
    progress = tolerance_engine.update_progress(
        "U001", week=2, comfort_level=4, dishes_tried=["CQ001", "CQ003"]
    )
    print(f"\nProgress update:")
    print(f"  Week: {progress.current_week}")
    print(f"  Tolerance gain: {progress.tolerance_gain:.2f}")
    print(f"  Comfort level: {progress.comfort_level}")
    print(f"  Confidence change: {progress.confidence_change:+.2f}")