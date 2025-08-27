import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import pickle
import logging
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from scent_rl_environment import ScentTherapyEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Experience tuple
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for scent therapy optimization
    
    Architecture:
    - Input: 27-dimensional state vector
    - Hidden layers: 3 fully connected layers with dropout
    - Output: Q-values for each action dimension
    """
    
    def __init__(self, state_size: int = 27, action_size: int = 15, 
                 hidden_size: int = 256, dropout: float = 0.3):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Separate heads for each action dimension (Dueling DQN concept)
        self.scent_type_head = nn.Linear(hidden_size // 2, 5)  # 5 scent types
        self.intensity_head = nn.Linear(hidden_size // 2, 5)   # 5 intensity levels
        self.duration_head = nn.Linear(hidden_size // 2, 4)    # 4 duration options
        
        # Value function head (for Dueling DQN)
        self.value_head = nn.Linear(hidden_size // 2, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """Forward pass through the network"""
        features = self.feature_net(state)
        
        # Get Q-values for each action dimension
        scent_q = self.scent_type_head(features)
        intensity_q = self.intensity_head(features)
        duration_q = self.duration_head(features)
        
        # Value function for Dueling DQN
        value = self.value_head(features)
        
        # Combine Q-values (simplified approach)
        # In practice, you might want to use a more sophisticated combination
        combined_q = torch.cat([scent_q, intensity_q, duration_q], dim=1)
        
        return combined_q, value, (scent_q, intensity_q, duration_q)


class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        """Add experience with maximum priority"""
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(self.max_priority)
        else:
            # Replace oldest experience
            idx = len(self.buffer) % self.capacity
            self.buffer[idx] = experience
            self.priorities[idx] = self.max_priority
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample batch with importance sampling"""
        if len(self.buffer) < batch_size:
            return [], [], []
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, td_errors):
            priority = abs(error) + 1e-6  # Small constant to avoid zero priority
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent for scent therapy optimization
    
    Features:
    - Double DQN to reduce overestimation
    - Dueling DQN architecture
    - Prioritized experience replay
    - Epsilon-greedy exploration with decay
    - Target network with soft updates
    - Multi-dimensional action space handling
    """
    
    def __init__(self, state_size: int = 27, config: Dict = None):
        self.state_size = state_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default configuration
        default_config = {
            'hidden_size': 256,
            'learning_rate': 0.001,
            'batch_size': 64,
            'buffer_capacity': 100000,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'target_update_freq': 1000,
            'tau': 0.005,  # Soft update parameter
            'gamma': 0.95,  # Discount factor
            'prioritized_replay': True,
            'double_dqn': True,
            'dueling_dqn': True
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Networks
        self.q_network = DQNNetwork(
            state_size=state_size,
            hidden_size=self.config['hidden_size']
        ).to(self.device)
        
        self.target_network = DQNNetwork(
            state_size=state_size, 
            hidden_size=self.config['hidden_size']
        ).to(self.device)
        
        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=50, verbose=True
        )
        
        # Experience replay
        if self.config['prioritized_replay']:
            self.memory = PrioritizedReplayBuffer(self.config['buffer_capacity'])
        else:
            self.memory = ReplayBuffer(self.config['buffer_capacity'])
        
        # Exploration parameters
        self.epsilon = self.config['epsilon_start']
        self.epsilon_end = self.config['epsilon_end']
        self.epsilon_decay = self.config['epsilon_decay']
        
        # Training tracking
        self.training_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.q_values = []
        
        logger.info(f"DQN Agent initialized on {self.device}")
        logger.info(f"Config: {self.config}")
    
    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using epsilon-greedy policy"""
        
        # Exploration
        if training and random.random() < self.epsilon:
            return self._random_action()
        
        # Exploitation
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values, _, (scent_q, intensity_q, duration_q) = self.q_network(state_tensor)
            
            # Select best action for each dimension
            scent_action = torch.argmax(scent_q, dim=1).item()
            intensity_action = torch.argmax(intensity_q, dim=1).item()
            duration_action = torch.argmax(duration_q, dim=1).item()
            
            # Store Q-values for monitoring
            if training:
                self.q_values.append(float(q_values.max()))
        
        # Convert discrete actions to continuous action space
        action = np.array([
            scent_action,
            intensity_action / 4.0,  # Normalize to [0, 1]
            duration_action
        ], dtype=np.float32)
        
        return action
    
    def _random_action(self) -> np.ndarray:
        """Generate random action"""
        return np.array([
            np.random.randint(0, 5),      # scent type
            np.random.random(),           # intensity [0, 1]
            np.random.randint(0, 4)       # duration
        ], dtype=np.float32)
    
    def step(self, state, action, reward, next_state, done):
        """Store experience and potentially learn"""
        self.memory.push(state, action, reward, next_state, done)
        
        # Learn if enough experiences
        if len(self.memory) >= self.config['batch_size']:
            self._learn()
    
    def _learn(self):
        """Update Q-network using batch of experiences"""
        if self.config['prioritized_replay']:
            experiences, indices, weights = self.memory.sample(
                self.config['batch_size']
            )
            if not experiences:
                return
                
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            experiences = self.memory.sample(self.config['batch_size'])
            weights = torch.ones(self.config['batch_size']).to(self.device)
            indices = None
        
        # Unpack experiences
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([self._discretize_action(e.action) for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q-values
        current_q_values, _, _ = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values
        with torch.no_grad():
            if self.config['double_dqn']:
                # Double DQN: use main network to select actions, target network to evaluate
                next_q_values_main, _, _ = self.q_network(next_states)
                next_actions = next_q_values_main.argmax(1).unsqueeze(1)
                
                next_q_values_target, _, _ = self.target_network(next_states)
                next_q_values = next_q_values_target.gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN
                next_q_values, _, _ = self.target_network(next_states)
                next_q_values = next_q_values.max(1)[0]
            
            target_q_values = rewards + (self.config['gamma'] * next_q_values * ~dones)
        
        # Compute loss
        td_errors = target_q_values - current_q_values
        
        if self.config['prioritized_replay']:
            loss = (weights * td_errors.pow(2)).mean()
            
            # Update priorities
            if indices is not None:
                self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.config['target_update_freq'] == 0:
            self._soft_update_target_network()
        
        # Update exploration rate
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
        
        # Track training metrics
        self.losses.append(float(loss))
        
        return float(loss)
    
    def _discretize_action(self, action: np.ndarray) -> int:
        """Convert continuous action to discrete action index"""
        scent_type = int(action[0])
        intensity_level = int(action[1] * 4)  # Convert to 0-4 range
        duration_idx = int(action[2])
        
        # Simple encoding: scent_type * 20 + intensity * 4 + duration
        return scent_type * 20 + intensity_level * 4 + duration_idx
    
    def _soft_update_target_network(self):
        """Soft update of target network parameters"""
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(
                self.config['tau'] * local_param.data + 
                (1.0 - self.config['tau']) * target_param.data
            )
    
    def train(self, env: ScentTherapyEnvironment, episodes: int = 1000,
              max_steps_per_episode: int = 100, save_freq: int = 100):
        """Train the DQN agent"""
        
        logger.info(f"Starting training for {episodes} episodes")
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps_per_episode):
                # Select action
                action = self.act(state, training=True)
                
                # Take step in environment
                next_state, reward, done, info = env.step(action)
                
                # Store experience and learn
                self.step(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Track episode metrics
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            # Update learning rate
            if len(self.episode_rewards) >= 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                self.scheduler.step(avg_reward)
            
            # Logging
            if episode % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else total_reward
                avg_loss = np.mean(self.losses[-100:]) if len(self.losses) >= 100 else 0
                avg_q = np.mean(self.q_values[-100:]) if len(self.q_values) >= 100 else 0
                
                logger.info(f"Episode {episode}: "
                          f"Avg Reward: {avg_reward:.2f}, "
                          f"Avg Loss: {avg_loss:.4f}, "
                          f"Avg Q: {avg_q:.2f}, "
                          f"Epsilon: {self.epsilon:.3f}")
            
            # Save model
            if episode % save_freq == 0 and episode > 0:
                self.save_model(f'dqn_checkpoint_episode_{episode}.pth')
        
        logger.info("Training completed")
        
        # Final save
        self.save_model('dqn_final_model.pth')
        
        # Plot training curves
        self.plot_training_progress()
    
    def evaluate(self, env: ScentTherapyEnvironment, episodes: int = 10) -> Dict:
        """Evaluate trained agent"""
        logger.info(f"Evaluating agent for {episodes} episodes")
        
        evaluation_rewards = []
        evaluation_stress_reductions = []
        evaluation_actions = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            initial_stress = env._calculate_stress_index()
            episode_actions = []
            
            for step in range(100):  # Max steps per episode
                action = self.act(state, training=False)
                next_state, reward, done, info = env.step(action)
                
                total_reward += reward
                episode_actions.append(info['scent_action'])
                state = next_state
                
                if done:
                    break
            
            final_stress = env._calculate_stress_index()
            stress_reduction = initial_stress - final_stress
            
            evaluation_rewards.append(total_reward)
            evaluation_stress_reductions.append(stress_reduction)
            evaluation_actions.extend(episode_actions)
        
        # Calculate statistics
        results = {
            'average_reward': float(np.mean(evaluation_rewards)),
            'std_reward': float(np.std(evaluation_rewards)),
            'average_stress_reduction': float(np.mean(evaluation_stress_reductions)),
            'std_stress_reduction': float(np.std(evaluation_stress_reductions)),
            'scent_usage': {
                scent: sum(1 for a in evaluation_actions if a.scent_type == scent)
                for scent in ['none', 'lavender', 'citrus', 'mint', 'eucalyptus']
            }
        }
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def save_model(self, filepath: str):
        """Save model and training state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.config = checkpoint.get('config', self.config)
        self.training_step = checkpoint.get('training_step', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.losses = checkpoint.get('losses', [])
        
        logger.info(f"Model loaded from {filepath}")
    
    def plot_training_progress(self):
        """Plot training progress"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        # Episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True)
        
        # Training loss
        if self.losses:
            ax3.plot(self.losses)
            ax3.set_title('Training Loss')
            ax3.set_xlabel('Training Step')
            ax3.set_ylabel('Loss')
            ax3.grid(True)
        
        # Average Q-values
        if self.q_values:
            ax4.plot(self.q_values)
            ax4.set_title('Average Q-Values')
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Q-Value')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('dqn_training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_action_preferences(self, test_states: List[np.ndarray]) -> Dict:
        """Analyze learned action preferences for different states"""
        self.q_network.eval()
        
        action_preferences = {
            'scent_types': [],
            'intensities': [],
            'durations': []
        }
        
        with torch.no_grad():
            for state in test_states:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                _, _, (scent_q, intensity_q, duration_q) = self.q_network(state_tensor)
                
                # Get preferred actions
                preferred_scent = torch.argmax(scent_q, dim=1).item()
                preferred_intensity = torch.argmax(intensity_q, dim=1).item()
                preferred_duration = torch.argmax(duration_q, dim=1).item()
                
                action_preferences['scent_types'].append(preferred_scent)
                action_preferences['intensities'].append(preferred_intensity)
                action_preferences['durations'].append(preferred_duration)
        
        return action_preferences


if __name__ == "__main__":
    # Example usage
    from scent_rl_environment import ScentTherapyEnvironment
    
    # Create environment
    env = ScentTherapyEnvironment("CQ_001")
    
    # Create agent
    agent = DQNAgent(state_size=27)
    
    # Train agent
    agent.train(env, episodes=100, max_steps_per_episode=50)
    
    # Evaluate agent
    results = agent.evaluate(env, episodes=5)
    print("Evaluation results:", json.dumps(results, indent=2))
    
    # Test action preferences
    test_states = [env.reset() for _ in range(10)]
    preferences = agent.get_action_preferences(test_states)
    print("Action preferences:", preferences)