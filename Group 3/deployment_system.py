import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import sqlite3
import threading
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

from emotion_physiology_model import EmotionPredictor
from dqn_agent import DQNAgent
from scent_rl_environment import ScentTherapyEnvironment
from data_collection import PhysiologicalReading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScentCommand:
    """Command to control scent release"""
    participant_id: str
    scent_type: str
    intensity: float
    duration: int
    timestamp: datetime
    confidence: float = 0.0


@dataclass
class SystemStatus:
    """Current system status"""
    participant_id: str
    is_active: bool
    current_emotion: str
    emotion_confidence: float
    stress_level: float
    last_scent: Optional[str]
    scent_count_today: int
    battery_level: float = 100.0
    connection_status: str = "connected"


class PhysiologicalData(BaseModel):
    """API model for physiological data input"""
    participant_id: str
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    hrv_rmssd: Optional[float] = None
    hrv_pnn50: Optional[float] = None
    skin_conductance: Optional[float] = None
    skin_temperature: Optional[float] = None
    location: str = "indoor"
    activity: str = "resting"
    emotion: Optional[str] = None
    notes: str = ""


class UserFeedback(BaseModel):
    """API model for user feedback"""
    participant_id: str
    feedback: str  # 'good', 'bad', 'better', 'worse', 'neutral'
    rating: Optional[int] = None  # 1-5 scale
    notes: str = ""


class ElderWellbeingSystem:
    """
    Complete deployment system for Elder Well-being AI
    
    Features:
    - Real-time physiological monitoring
    - Emotion prediction and tracking
    - Intelligent scent therapy recommendations
    - User feedback integration
    - Safety monitoring and alerts
    - Data logging and analytics
    - Web dashboard for monitoring
    """
    
    def __init__(self, config_path: str = "deployment_config.json"):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.emotion_predictor = None
        self.agents = {}
        self.environments = {}
        
        # System state
        self.active_participants = {}
        self.system_status = {}
        self.command_queue = Queue()
        self.data_buffer = {}
        
        # Safety monitoring
        self.safety_alerts = {}
        self.emergency_stops = set()
        
        # Logging
        self.db_connection = None
        
        # WebSocket connections
        self.websocket_connections = []
        
        # Initialize system
        self._initialize_system()
        
        # Start background tasks
        self.monitoring_task = None
        self.is_running = False
        
        logger.info("Elder Well-being System initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load deployment configuration"""
        default_config = {
            "models": {
                "emotion_model_path": "results/emotion_model.pth",
                "agent_models_path": "results/",
                "participants": ["CQ_001", "CQ_002", "CQ_003"]
            },
            "safety": {
                "max_scents_per_hour": 6,
                "max_scents_per_day": 30,
                "min_interval_seconds": 300,
                "emergency_hr_threshold": 180,
                "emergency_bp_threshold": 200,
                "stress_alert_threshold": 0.9
            },
            "monitoring": {
                "data_collection_interval": 30,  # seconds
                "prediction_interval": 60,
                "safety_check_interval": 10,
                "auto_scent_enabled": True
            },
            "api": {
                "host": "localhost",
                "port": 8000,
                "cors_origins": ["*"]
            },
            "hardware": {
                "scent_device_interface": "mock",  # 'mock' or 'serial' or 'gpio'
                "sensor_interface": "mock"
            },
            "database": {
                "path": "deployment_data.db",
                "backup_interval": 3600
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Merge configurations
            config = {**default_config}
            for key, value in user_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
                    
            logger.info(f"Configuration loaded from {config_path}")
        except FileNotFoundError:
            config = default_config
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("Default configuration created")
        
        return config
    
    def _initialize_system(self):
        """Initialize all system components"""
        
        # Load emotion prediction model
        try:
            self.emotion_predictor = EmotionPredictor()
            self.emotion_predictor.load_model(self.config['models']['emotion_model_path'])
            logger.info("Emotion prediction model loaded")
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            self.emotion_predictor = None
        
        # Load DQN agents for each participant
        for participant_id in self.config['models']['participants']:
            try:
                agent_path = f"{self.config['models']['agent_models_path']}dqn_agent_{participant_id}.pth"
                agent = DQNAgent(state_size=27)
                agent.load_model(agent_path)
                self.agents[participant_id] = agent
                
                # Initialize environment
                env = ScentTherapyEnvironment(
                    participant_id=participant_id,
                    emotion_model_path=self.config['models']['emotion_model_path']
                )
                self.environments[participant_id] = env
                
                # Initialize status
                self.system_status[participant_id] = SystemStatus(
                    participant_id=participant_id,
                    is_active=False,
                    current_emotion="neutral",
                    emotion_confidence=0.5,
                    stress_level=0.5,
                    last_scent=None,
                    scent_count_today=0
                )
                
                self.data_buffer[participant_id] = []
                
                logger.info(f"Agent loaded for participant {participant_id}")
                
            except Exception as e:
                logger.error(f"Failed to load agent for {participant_id}: {e}")
        
        # Initialize database
        self._initialize_database()
        
        # Initialize hardware interfaces
        self._initialize_hardware()
    
    def _initialize_database(self):
        """Initialize deployment database"""
        try:
            self.db_connection = sqlite3.connect(
                self.config['database']['path'],
                check_same_thread=False
            )
            
            cursor = self.db_connection.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS deployment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    participant_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    heart_rate REAL,
                    systolic_bp REAL,
                    diastolic_bp REAL,
                    predicted_emotion TEXT,
                    emotion_confidence REAL,
                    stress_level REAL,
                    scent_command TEXT,
                    user_feedback TEXT,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scent_commands (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    participant_id TEXT NOT NULL,
                    scent_type TEXT NOT NULL,
                    intensity REAL NOT NULL,
                    duration INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    confidence REAL,
                    executed BOOLEAN DEFAULT FALSE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS safety_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    participant_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT,
                    physiological_data TEXT,
                    action_taken TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.db_connection.commit()
            logger.info("Database initialized")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def _initialize_hardware(self):
        """Initialize hardware interfaces"""
        # This is a mock implementation
        # In production, this would interface with actual hardware
        
        if self.config['hardware']['scent_device_interface'] == 'mock':
            logger.info("Mock scent device interface initialized")
        
        if self.config['hardware']['sensor_interface'] == 'mock':
            logger.info("Mock sensor interface initialized")
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        self.is_running = True
        
        # Start background monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Process each active participant
                for participant_id in self.active_participants:
                    await self._process_participant(participant_id)
                
                # Process command queue
                await self._process_commands()
                
                # Safety checks
                await self._safety_monitoring()
                
                # Notify connected clients
                await self._notify_clients()
                
                # Sleep for monitoring interval
                await asyncio.sleep(self.config['monitoring']['data_collection_interval'])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _process_participant(self, participant_id: str):
        """Process data and generate recommendations for a participant"""
        
        if participant_id not in self.data_buffer or not self.data_buffer[participant_id]:
            return
        
        # Get recent data
        recent_data = self.data_buffer[participant_id][-10:]  # Last 10 readings
        
        if len(recent_data) < 3:
            return  # Need minimum data for prediction
        
        try:
            # Prepare data for emotion prediction
            physiological_sequence = np.array([[
                reading.heart_rate,
                reading.systolic_bp,
                reading.diastolic_bp,
                reading.hrv_rmssd or 25.0,
                reading.skin_conductance or 2.0,
                reading.skin_temperature or 36.5
            ] + [0] * 12 for reading in recent_data])  # Pad with zeros
            
            # Predict emotion
            if self.emotion_predictor:
                emotion_result = self.emotion_predictor.predict_emotion(physiological_sequence)
                
                predicted_emotion = emotion_result['predicted_emotion']
                emotion_confidence = emotion_result['confidence']
                stress_level = emotion_result.get('physiological_trends', {}).get('stress_indicators', {}).get('high_hr', 0.5)
            else:
                # Fallback emotion prediction
                predicted_emotion = 'neutral'
                emotion_confidence = 0.5
                stress_level = 0.5
            
            # Update system status
            self.system_status[participant_id].current_emotion = predicted_emotion
            self.system_status[participant_id].emotion_confidence = emotion_confidence
            self.system_status[participant_id].stress_level = stress_level
            
            # Generate scent recommendation using RL agent
            if (participant_id in self.agents and 
                self.config['monitoring']['auto_scent_enabled']):
                
                await self._generate_scent_recommendation(participant_id)
            
            # Log data
            self._log_participant_data(participant_id, recent_data[-1], 
                                     predicted_emotion, emotion_confidence, stress_level)
            
        except Exception as e:
            logger.error(f"Error processing participant {participant_id}: {e}")
    
    async def _generate_scent_recommendation(self, participant_id: str):
        """Generate scent recommendation using RL agent"""
        
        try:
            env = self.environments[participant_id]
            agent = self.agents[participant_id]
            
            # Get current state from environment
            # This is simplified - in practice, you'd update the environment state
            # with the latest physiological data
            current_state = env._get_state_vector()
            
            # Get action recommendation
            action = agent.act(current_state, training=False)
            
            # Parse action
            scent_action = env._parse_action(action)
            
            # Check if scent should be recommended
            if (scent_action.scent_type != 'none' and 
                self._should_recommend_scent(participant_id, scent_action)):
                
                # Create scent command
                command = ScentCommand(
                    participant_id=participant_id,
                    scent_type=scent_action.scent_type,
                    intensity=scent_action.intensity,
                    duration=scent_action.duration,
                    timestamp=datetime.now(),
                    confidence=0.8  # Could be derived from model uncertainty
                )
                
                # Add to command queue
                self.command_queue.put(command)
                
                logger.info(f"Scent recommendation for {participant_id}: "
                          f"{scent_action.scent_type} at {scent_action.intensity:.1f} intensity")
        
        except Exception as e:
            logger.error(f"Error generating scent recommendation for {participant_id}: {e}")
    
    def _should_recommend_scent(self, participant_id: str, scent_action) -> bool:
        """Check if scent should be recommended based on safety constraints"""
        
        status = self.system_status[participant_id]
        
        # Check daily limits
        if status.scent_count_today >= self.config['safety']['max_scents_per_day']:
            return False
        
        # Check minimum interval
        if status.last_scent:
            # This would check the actual time since last scent
            # For now, simplified check
            pass
        
        # Check stress level (don't recommend if very low stress)
        if status.stress_level < 0.2:
            return False
        
        # Check emergency conditions
        if participant_id in self.emergency_stops:
            return False
        
        return True
    
    async def _process_commands(self):
        """Process queued scent commands"""
        processed_commands = []
        
        while not self.command_queue.empty():
            command = self.command_queue.get()
            
            try:
                # Execute scent command
                success = await self._execute_scent_command(command)
                
                if success:
                    # Update system status
                    self.system_status[command.participant_id].last_scent = command.scent_type
                    self.system_status[command.participant_id].scent_count_today += 1
                    
                    # Log command
                    self._log_scent_command(command, executed=True)
                    
                    processed_commands.append(command)
                    
                    logger.info(f"Executed scent command: {command.scent_type} "
                              f"for {command.participant_id}")
                else:
                    logger.warning(f"Failed to execute scent command for {command.participant_id}")
            
            except Exception as e:
                logger.error(f"Error executing command: {e}")
    
    async def _execute_scent_command(self, command: ScentCommand) -> bool:
        """Execute scent release command"""
        
        # Mock implementation - in production, this would control actual hardware
        if self.config['hardware']['scent_device_interface'] == 'mock':
            # Simulate scent release
            await asyncio.sleep(0.1)
            logger.info(f"Mock scent release: {command.scent_type} "
                       f"at {command.intensity:.1f} intensity for {command.duration}s")
            return True
        else:
            # Real hardware interface would go here
            return False
    
    async def _safety_monitoring(self):
        """Monitor for safety conditions"""
        
        for participant_id, status in self.system_status.items():
            
            # Check for emergency physiological conditions
            if participant_id in self.data_buffer and self.data_buffer[participant_id]:
                latest_reading = self.data_buffer[participant_id][-1]
                
                emergency_conditions = []
                
                # Check heart rate
                if latest_reading.heart_rate > self.config['safety']['emergency_hr_threshold']:
                    emergency_conditions.append(f"High heart rate: {latest_reading.heart_rate}")
                
                # Check blood pressure
                if latest_reading.systolic_bp > self.config['safety']['emergency_bp_threshold']:
                    emergency_conditions.append(f"High blood pressure: {latest_reading.systolic_bp}")
                
                # Check stress level
                if status.stress_level > self.config['safety']['stress_alert_threshold']:
                    emergency_conditions.append(f"High stress level: {status.stress_level:.2f}")
                
                # Handle emergency conditions
                if emergency_conditions:
                    await self._handle_safety_alert(participant_id, emergency_conditions, latest_reading)
    
    async def _handle_safety_alert(self, participant_id: str, conditions: List[str], 
                                 reading: PhysiologicalReading):
        """Handle safety alert"""
        
        alert_key = f"{participant_id}_{datetime.now().strftime('%Y%m%d_%H')}"
        
        # Avoid duplicate alerts within the same hour
        if alert_key in self.safety_alerts:
            return
        
        self.safety_alerts[alert_key] = {
            'participant_id': participant_id,
            'conditions': conditions,
            'timestamp': datetime.now(),
            'severity': 'high' if len(conditions) > 1 else 'medium'
        }
        
        # Stop automatic scent recommendations temporarily
        self.emergency_stops.add(participant_id)
        
        # Log safety event
        self._log_safety_event(participant_id, conditions, reading)
        
        # Notify connected clients
        await self._notify_safety_alert(participant_id, conditions)
        
        logger.warning(f"Safety alert for {participant_id}: {', '.join(conditions)}")
    
    def _log_participant_data(self, participant_id: str, reading: PhysiologicalReading,
                            emotion: str, confidence: float, stress_level: float):
        """Log participant data to database"""
        
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO deployment_data 
                (participant_id, timestamp, heart_rate, systolic_bp, diastolic_bp,
                 predicted_emotion, emotion_confidence, stress_level, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                participant_id, reading.timestamp.isoformat(),
                reading.heart_rate, reading.systolic_bp, reading.diastolic_bp,
                emotion, confidence, stress_level, reading.notes
            ))
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error logging participant data: {e}")
    
    def _log_scent_command(self, command: ScentCommand, executed: bool):
        """Log scent command to database"""
        
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO scent_commands
                (participant_id, scent_type, intensity, duration, timestamp, confidence, executed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                command.participant_id, command.scent_type, command.intensity,
                command.duration, command.timestamp.isoformat(),
                command.confidence, executed
            ))
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error logging scent command: {e}")
    
    def _log_safety_event(self, participant_id: str, conditions: List[str], 
                         reading: PhysiologicalReading):
        """Log safety event to database"""
        
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO safety_events
                (participant_id, event_type, severity, description, physiological_data, 
                 action_taken, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                participant_id, 'emergency_physiological', 'high',
                '; '.join(conditions), json.dumps(asdict(reading), default=str),
                'stopped_auto_scents', datetime.now().isoformat()
            ))
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error logging safety event: {e}")
    
    async def _notify_clients(self):
        """Notify connected WebSocket clients of updates"""
        
        if not self.websocket_connections:
            return
        
        # Prepare status update
        status_update = {
            'type': 'status_update',
            'timestamp': datetime.now().isoformat(),
            'participants': {
                pid: asdict(status) for pid, status in self.system_status.items()
                if pid in self.active_participants
            }
        }
        
        # Send to all connected clients
        disconnected_clients = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(status_update, default=str))
            except:
                disconnected_clients.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected_clients:
            self.websocket_connections.remove(websocket)
    
    async def _notify_safety_alert(self, participant_id: str, conditions: List[str]):
        """Send safety alert to connected clients"""
        
        alert_message = {
            'type': 'safety_alert',
            'participant_id': participant_id,
            'conditions': conditions,
            'timestamp': datetime.now().isoformat(),
            'severity': 'high' if len(conditions) > 1 else 'medium'
        }
        
        # Send to all connected clients
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(alert_message, default=str))
            except:
                pass
    
    # API Methods
    
    def activate_participant(self, participant_id: str) -> bool:
        """Activate monitoring for a participant"""
        
        if participant_id not in self.system_status:
            logger.error(f"Participant {participant_id} not found in system")
            return False
        
        self.active_participants[participant_id] = True
        self.system_status[participant_id].is_active = True
        
        # Clear emergency stop if set
        if participant_id in self.emergency_stops:
            self.emergency_stops.remove(participant_id)
        
        logger.info(f"Activated monitoring for participant {participant_id}")
        return True
    
    def deactivate_participant(self, participant_id: str) -> bool:
        """Deactivate monitoring for a participant"""
        
        if participant_id in self.active_participants:
            del self.active_participants[participant_id]
        
        if participant_id in self.system_status:
            self.system_status[participant_id].is_active = False
        
        logger.info(f"Deactivated monitoring for participant {participant_id}")
        return True
    
    def add_physiological_data(self, data: PhysiologicalData) -> bool:
        """Add physiological data for processing"""
        
        try:
            # Convert to PhysiologicalReading
            reading = PhysiologicalReading(
                participant_id=data.participant_id,
                timestamp=datetime.now(),
                heart_rate=data.heart_rate,
                systolic_bp=data.systolic_bp,
                diastolic_bp=data.diastolic_bp,
                hrv_rmssd=data.hrv_rmssd,
                hrv_pnn50=data.hrv_pnn50,
                skin_conductance=data.skin_conductance,
                skin_temperature=data.skin_temperature,
                location=data.location,
                activity=data.activity,
                emotion=data.emotion or "unknown",
                notes=data.notes
            )
            
            # Add to buffer
            if data.participant_id not in self.data_buffer:
                self.data_buffer[data.participant_id] = []
            
            self.data_buffer[data.participant_id].append(reading)
            
            # Keep only recent data (last 100 readings)
            if len(self.data_buffer[data.participant_id]) > 100:
                self.data_buffer[data.participant_id] = self.data_buffer[data.participant_id][-100:]
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding physiological data: {e}")
            return False
    
    def add_user_feedback(self, feedback: UserFeedback) -> bool:
        """Add user feedback for system learning"""
        
        try:
            # Update RL environment with feedback
            if feedback.participant_id in self.environments:
                env = self.environments[feedback.participant_id]
                reward_adjustment = env.get_user_feedback(feedback.feedback)
                
                logger.info(f"User feedback for {feedback.participant_id}: "
                          f"{feedback.feedback} (reward: {reward_adjustment})")
            
            # Log feedback to database
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO deployment_data 
                (participant_id, timestamp, user_feedback, notes)
                VALUES (?, ?, ?, ?)
            ''', (
                feedback.participant_id, datetime.now().isoformat(),
                json.dumps(asdict(feedback)), feedback.notes
            ))
            self.db_connection.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding user feedback: {e}")
            return False
    
    def get_participant_status(self, participant_id: str) -> Optional[SystemStatus]:
        """Get current status for a participant"""
        return self.system_status.get(participant_id)
    
    def get_system_overview(self) -> Dict:
        """Get system overview"""
        return {
            'total_participants': len(self.system_status),
            'active_participants': len(self.active_participants),
            'safety_alerts': len(self.safety_alerts),
            'system_uptime': str(datetime.now() - datetime.now()),  # Placeholder
            'is_monitoring': self.is_running,
            'participants': {
                pid: asdict(status) for pid, status in self.system_status.items()
            }
        }


# FastAPI application
app = FastAPI(title="Elder Well-being AI System", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance
system = ElderWellbeingSystem()


@app.on_event("startup")
async def startup_event():
    """Start the monitoring system"""
    await system.start_monitoring()


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the monitoring system"""
    await system.stop_monitoring()


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Elder Well-being AI System", "status": "running"}


@app.get("/system/overview")
async def get_system_overview():
    """Get system overview"""
    return system.get_system_overview()


@app.post("/participants/{participant_id}/activate")
async def activate_participant(participant_id: str):
    """Activate monitoring for a participant"""
    success = system.activate_participant(participant_id)
    if success:
        return {"message": f"Participant {participant_id} activated"}
    else:
        raise HTTPException(status_code=404, detail="Participant not found")


@app.post("/participants/{participant_id}/deactivate")
async def deactivate_participant(participant_id: str):
    """Deactivate monitoring for a participant"""
    success = system.deactivate_participant(participant_id)
    return {"message": f"Participant {participant_id} deactivated"}


@app.get("/participants/{participant_id}/status")
async def get_participant_status(participant_id: str):
    """Get participant status"""
    status = system.get_participant_status(participant_id)
    if status:
        return asdict(status)
    else:
        raise HTTPException(status_code=404, detail="Participant not found")


@app.post("/data/physiological")
async def add_physiological_data(data: PhysiologicalData):
    """Add physiological data"""
    success = system.add_physiological_data(data)
    if success:
        return {"message": "Data added successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to add data")


@app.post("/feedback/user")
async def add_user_feedback(feedback: UserFeedback):
    """Add user feedback"""
    success = system.add_user_feedback(feedback)
    if success:
        return {"message": "Feedback added successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to add feedback")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    system.websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        system.websocket_connections.remove(websocket)


@app.get("/dashboard")
async def get_dashboard():
    """Simple dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Elder Well-being AI Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .participant { border: 1px solid #ccc; padding: 10px; margin: 10px 0; }
            .status { padding: 5px; margin: 5px 0; }
            .active { background-color: #d4edda; }
            .inactive { background-color: #f8d7da; }
            .alert { background-color: #fff3cd; }
        </style>
    </head>
    <body>
        <h1>Elder Well-being AI Dashboard</h1>
        <div id="system-status"></div>
        <div id="participants"></div>
        
        <script>
            const ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'status_update') {
                    updateDashboard(data.participants);
                } else if (data.type === 'safety_alert') {
                    showAlert(data);
                }
            };
            
            function updateDashboard(participants) {
                const container = document.getElementById('participants');
                container.innerHTML = '';
                
                for (const [id, status] of Object.entries(participants)) {
                    const div = document.createElement('div');
                    div.className = 'participant';
                    div.innerHTML = `
                        <h3>Participant ${id}</h3>
                        <div class="status ${status.is_active ? 'active' : 'inactive'}">
                            Status: ${status.is_active ? 'Active' : 'Inactive'}
                        </div>
                        <div>Emotion: ${status.current_emotion} (${Math.round(status.emotion_confidence * 100)}%)</div>
                        <div>Stress Level: ${Math.round(status.stress_level * 100)}%</div>
                        <div>Last Scent: ${status.last_scent || 'None'}</div>
                        <div>Today's Scents: ${status.scent_count_today}</div>
                    `;
                    container.appendChild(div);
                }
            }
            
            function showAlert(alert) {
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert';
                alertDiv.innerHTML = `
                    <strong>Safety Alert for ${alert.participant_id}:</strong><br>
                    ${alert.conditions.join(', ')}<br>
                    <small>${alert.timestamp}</small>
                `;
                document.body.insertBefore(alertDiv, document.body.firstChild);
                
                setTimeout(() => alertDiv.remove(), 10000);
            }
            
            // Initial load
            fetch('/system/overview')
                .then(response => response.json())
                .then(data => updateDashboard(data.participants));
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)