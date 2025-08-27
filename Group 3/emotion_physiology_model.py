import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import json
import pickle
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhysiologicalDataset(Dataset):
    """
    Dataset for physiological signals to emotion mapping
    
    Features:
    - Heart Rate (HR)
    - Blood Pressure (Systolic/Diastolic) 
    - Heart Rate Variability (HRV)
    - Skin Conductance (optional)
    - Skin Temperature (optional)
    - Environmental context (time, location, weather)
    """
    
    def __init__(self, data: pd.DataFrame, sequence_length: int = 60, 
                 target_col: str = 'emotion', features: List[str] = None):
        self.data = data.sort_values(['participant_id', 'timestamp'])
        self.sequence_length = sequence_length
        self.target_col = target_col
        
        # Define physiological features
        if features is None:
            self.features = [
                'heart_rate', 'systolic_bp', 'diastolic_bp', 'hrv_rmssd',
                'hrv_pnn50', 'skin_conductance', 'skin_temperature',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos',  # time features
                'location_indoor', 'location_outdoor', 'weather_sunny', 
                'weather_cloudy', 'weather_rainy', 'activity_resting',
                'activity_walking', 'activity_social'
            ]
        else:
            self.features = features
        
        # Prepare sequences
        self.sequences, self.labels = self._create_sequences()
        
        # Normalize features
        self.scaler = StandardScaler()
        self.sequences = self.scaler.fit_transform(
            self.sequences.reshape(-1, len(self.features))
        ).reshape(self.sequences.shape)
        
        logger.info(f"Created {len(self.sequences)} sequences with length {sequence_length}")
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences from time series data"""
        sequences = []
        labels = []
        
        # Group by participant to avoid mixing data across participants
        for participant_id in self.data['participant_id'].unique():
            participant_data = self.data[
                self.data['participant_id'] == participant_id
            ].reset_index(drop=True)
            
            # Create sequences for this participant
            for i in range(len(participant_data) - self.sequence_length + 1):
                # Extract sequence of physiological features
                sequence = participant_data.iloc[i:i+self.sequence_length][self.features].values
                
                # Get emotion label (use the last timestamp in sequence)
                label = participant_data.iloc[i+self.sequence_length-1][self.target_col]
                
                sequences.append(sequence)
                labels.append(label)
        
        return np.array(sequences), np.array(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.LongTensor([self.labels[idx]])[0]
        )


class EmotionPhysiologyLSTM(nn.Module):
    """
    LSTM model for mapping physiological signals to emotional states
    
    Architecture:
    - Input: Sequential physiological data
    - LSTM layers: Extract temporal patterns
    - Attention: Focus on important time steps
    - Output: Probability distribution over emotion classes
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, num_classes: int = 6, 
                 dropout: float = 0.3):
        super(EmotionPhysiologyLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attended_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Use mean pooling over sequence dimension
        pooled_out = torch.mean(attended_out, dim=1)
        
        # Classification
        emotion_logits = self.classifier(pooled_out)
        confidence = self.confidence_head(pooled_out)
        
        return {
            'logits': emotion_logits,
            'confidence': confidence,
            'attention_weights': attention_weights,
            'hidden_states': pooled_out
        }


class EmotionPredictor:
    """Main class for emotion prediction from physiological data"""
    
    def __init__(self, model_config: Dict = None):
        self.model_config = model_config or {
            'input_dim': 18,  # number of physiological features
            'hidden_dim': 128,
            'num_layers': 2,
            'num_classes': 6,  # calm, anxious, sad, happy, stressed, neutral
            'dropout': 0.3,
            'sequence_length': 60  # 60 seconds of data
        }
        
        # Emotion classes
        self.emotion_classes = [
            'calm', 'anxious', 'sad', 'happy', 'stressed', 'neutral'
        ]
        
        # Initialize model
        self.model = EmotionPhysiologyLSTM(**self.model_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.label_encoder = LabelEncoder()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def prepare_data(self, data_path: str, test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data"""
        
        # Load data
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            data = pd.read_json(data_path)
        else:
            raise ValueError("Data file must be CSV or JSON format")
        
        # Encode emotion labels
        data['emotion_encoded'] = self.label_encoder.fit_transform(data['emotion'])
        
        # Split by participants to avoid data leakage
        participants = data['participant_id'].unique()
        train_participants, val_participants = train_test_split(
            participants, test_size=test_size, random_state=42
        )
        
        train_data = data[data['participant_id'].isin(train_participants)]
        val_data = data[data['participant_id'].isin(val_participants)]
        
        # Create datasets
        train_dataset = PhysiologicalDataset(
            train_data, 
            sequence_length=self.model_config['sequence_length'],
            target_col='emotion_encoded'
        )
        
        val_dataset = PhysiologicalDataset(
            val_data,
            sequence_length=self.model_config['sequence_length'], 
            target_col='emotion_encoded'
        )
        
        # Store scaler for inference
        self.scaler = train_dataset.scaler
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, 
            num_workers=4, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, lr: float = 0.001):
        """Train the emotion prediction model"""
        
        # Setup training
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        
        logger.info(f"Starting training for {epochs} epochs on {self.device}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(data)
                loss = criterion(outputs['logits'], targets)
                
                # Add confidence regularization
                confidence_loss = torch.mean((outputs['confidence'] - 0.8) ** 2)
                total_loss = loss + 0.1 * confidence_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += total_loss.item()
                _, predicted = outputs['logits'].max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            val_loss, val_acc = self._validate(val_loader, criterion)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_emotion_model.pth')
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                          f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {val_acc:.2f}%")
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    def _validate(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = criterion(outputs['logits'], targets)
                
                val_loss += loss.item()
                _, predicted = outputs['logits'].max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        return val_loss, val_acc
    
    def predict_emotion(self, physiological_data: np.ndarray) -> Dict:
        """
        Predict emotion from physiological data
        
        Args:
            physiological_data: Shape (sequence_length, num_features)
        
        Returns:
            Dictionary with predictions and confidence
        """
        self.model.eval()
        
        # Normalize data
        if self.scaler is not None:
            physiological_data = self.scaler.transform(physiological_data)
        
        # Convert to tensor
        data_tensor = torch.FloatTensor(physiological_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(data_tensor)
            
            # Get predictions
            probabilities = torch.softmax(outputs['logits'], dim=1)[0].cpu().numpy()
            predicted_class = np.argmax(probabilities)
            confidence = outputs['confidence'][0].cpu().numpy()[0]
            
            # Convert to emotion label
            predicted_emotion = self.emotion_classes[predicted_class]
            
            result = {
                'predicted_emotion': predicted_emotion,
                'confidence': float(confidence),
                'probabilities': {
                    emotion: float(prob) 
                    for emotion, prob in zip(self.emotion_classes, probabilities)
                },
                'physiological_trends': self._analyze_trends(physiological_data)
            }
        
        return result
    
    def _analyze_trends(self, data: np.ndarray) -> Dict:
        """Analyze physiological trends"""
        if len(data) < 10:
            return {}
        
        # Assuming first few columns are HR, BP, HRV
        hr_trend = "increasing" if data[-5:, 0].mean() > data[:5, 0].mean() else "decreasing"
        bp_trend = "increasing" if data[-5:, 1].mean() > data[:5, 1].mean() else "decreasing"
        
        return {
            'heart_rate_trend': hr_trend,
            'blood_pressure_trend': bp_trend,
            'stress_indicators': {
                'high_hr': float(data[:, 0].mean() > 100),
                'high_bp': float(data[:, 1].mean() > 140),
                'low_hrv': float(data[:, 3].mean() < 20) if data.shape[1] > 3 else 0.0
            }
        }
    
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'emotion_classes': self.emotion_classes,
            'training_history': self.training_history
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model_config = checkpoint['model_config']
        self.model = EmotionPhysiologyLSTM(**self.model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        self.scaler = checkpoint['scaler']
        self.label_encoder = checkpoint['label_encoder']
        self.emotion_classes = checkpoint['emotion_classes']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.training_history['train_loss'], label='Train Loss')
        ax1.plot(self.training_history['val_loss'], label='Val Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.training_history['train_acc'], label='Train Acc')
        ax2.plot(self.training_history['val_acc'], label='Val Acc')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Example usage
    predictor = EmotionPredictor()
    
    # Note: You need to provide actual data file
    # train_loader, val_loader = predictor.prepare_data('physiological_data.csv')
    # predictor.train(train_loader, val_loader, epochs=100)
    # predictor.plot_training_history()
    
    print("Emotion-Physiology Model initialized successfully!")
    print("Available emotion classes:", predictor.emotion_classes)
    print("Model architecture:", predictor.model)