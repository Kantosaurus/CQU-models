import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import os
import logging
from datetime import datetime, timedelta
import argparse
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_collection import PhysiologicalDataCollector, DataSimulator
from emotion_physiology_model import EmotionPredictor
from scent_rl_environment import ScentTherapyEnvironment
from dqn_agent import DQNAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline for the Elder Well-being AI system
    
    Pipeline stages:
    1. Data Collection & Simulation
    2. Emotion-Physiology Model Training
    3. Reinforcement Learning Environment Setup
    4. DQN Agent Training for Scent Optimization
    5. Model Evaluation and Analysis
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_collector = None
        self.emotion_predictor = None
        self.environments = {}
        self.agents = {}
        
        # Training history
        self.training_history = {
            'emotion_model': {},
            'rl_agents': {}
        }
        
        logger.info("Training pipeline initialized")
        logger.info(f"Config: {json.dumps(self.config, indent=2)}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file or use defaults"""
        default_config = {
            'results_dir': 'results',
            'data': {
                'db_path': 'physiological_data.db',
                'participants': ['CQ_001', 'CQ_002', 'CQ_003', 'CQ_004', 'CQ_005'],
                'simulation_days': 21,
                'readings_per_day': 48,
                'export_path': 'training_data.csv'
            },
            'emotion_model': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'sequence_length': 60,
                'test_size': 0.2,
                'model_path': 'emotion_model.pth'
            },
            'rl_training': {
                'episodes': 2000,
                'max_steps_per_episode': 100,
                'evaluation_episodes': 50,
                'save_frequency': 200,
                'agent_config': {
                    'hidden_size': 256,
                    'learning_rate': 0.0005,
                    'batch_size': 64,
                    'buffer_capacity': 100000,
                    'epsilon_decay': 0.9995,
                    'target_update_freq': 1000,
                    'gamma': 0.95,
                    'prioritized_replay': True,
                    'double_dqn': True
                }
            },
            'evaluation': {
                'test_episodes': 100,
                'metrics': ['reward', 'stress_reduction', 'scent_efficiency', 'user_satisfaction']
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Merge configurations (user config overrides defaults)
            config = {**default_config}
            for key, value in user_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
            
            logger.info(f"Configuration loaded from {config_path}")
        else:
            config = default_config
            logger.info("Using default configuration")
        
        return config
    
    def stage1_data_collection(self):
        """Stage 1: Data Collection and Simulation"""
        logger.info("=== Stage 1: Data Collection and Simulation ===")
        
        # Initialize data collector
        self.data_collector = PhysiologicalDataCollector(
            db_path=self.config['data']['db_path']
        )
        
        # Initialize simulator
        simulator = DataSimulator(self.data_collector)
        
        # Simulate data for each participant
        for participant_id in self.config['data']['participants']:
            logger.info(f"Simulating data for participant {participant_id}")
            
            simulator.simulate_participant_data(
                participant_id=participant_id,
                days=self.config['data']['simulation_days'],
                readings_per_day=self.config['data']['readings_per_day']
            )
            
            # Generate individual reports
            report = self.data_collector.generate_summary_report(participant_id)
            
            # Save report
            report_path = self.results_dir / f"participant_{participant_id}_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Generate trend plots
            self.data_collector.plot_participant_trends(
                participant_id=participant_id,
                save_path=str(self.results_dir / f"trends_{participant_id}.png")
            )
        
        # Export training data
        logger.info("Exporting training data")
        training_data = self.data_collector.export_for_training(
            output_path=self.config['data']['export_path']
        )
        
        logger.info(f"Stage 1 completed. Generated {len(training_data)} training samples.")
        
        return training_data
    
    def stage2_emotion_model_training(self, training_data: pd.DataFrame = None):
        """Stage 2: Train Emotion-Physiology Mapping Model"""
        logger.info("=== Stage 2: Emotion-Physiology Model Training ===")
        
        # Load training data if not provided
        if training_data is None:
            if os.path.exists(self.config['data']['export_path']):
                training_data = pd.read_csv(self.config['data']['export_path'])
            else:
                logger.error("Training data not found. Run stage 1 first.")
                return None
        
        # Initialize emotion predictor
        self.emotion_predictor = EmotionPredictor()
        
        # Prepare data loaders
        train_loader, val_loader = self.emotion_predictor.prepare_data(
            data_path=self.config['data']['export_path'],
            test_size=self.config['emotion_model']['test_size']
        )
        
        # Train model
        self.emotion_predictor.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config['emotion_model']['epochs'],
            lr=self.config['emotion_model']['learning_rate']
        )
        
        # Save model
        model_path = self.results_dir / self.config['emotion_model']['model_path']
        self.emotion_predictor.save_model(str(model_path))
        
        # Plot training history
        self.emotion_predictor.plot_training_history()
        plt.savefig(self.results_dir / 'emotion_model_training.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store training history
        self.training_history['emotion_model'] = self.emotion_predictor.training_history
        
        logger.info("Stage 2 completed. Emotion model trained and saved.")
        
        return self.emotion_predictor
    
    def stage3_rl_setup(self):
        """Stage 3: Setup Reinforcement Learning Environments"""
        logger.info("=== Stage 3: RL Environment Setup ===")
        
        # Get emotion model path
        emotion_model_path = self.results_dir / self.config['emotion_model']['model_path']
        
        if not emotion_model_path.exists():
            logger.warning("Emotion model not found. RL training may be less accurate.")
            emotion_model_path = None
        
        # Create environments for each participant
        for participant_id in self.config['data']['participants']:
            logger.info(f"Setting up environment for {participant_id}")
            
            env = ScentTherapyEnvironment(
                participant_id=participant_id,
                emotion_model_path=str(emotion_model_path) if emotion_model_path else None,
                max_steps=self.config['rl_training']['max_steps_per_episode']
            )
            
            self.environments[participant_id] = env
        
        logger.info(f"Stage 3 completed. {len(self.environments)} environments created.")
    
    def stage4_rl_training(self):
        """Stage 4: Train DQN Agents for Scent Optimization"""
        logger.info("=== Stage 4: DQN Agent Training ===")
        
        # Train agent for each participant
        for participant_id, env in self.environments.items():
            logger.info(f"Training DQN agent for participant {participant_id}")
            
            # Create agent
            agent = DQNAgent(
                state_size=27,
                config=self.config['rl_training']['agent_config']
            )
            
            # Train agent
            agent.train(
                env=env,
                episodes=self.config['rl_training']['episodes'],
                max_steps_per_episode=self.config['rl_training']['max_steps_per_episode'],
                save_freq=self.config['rl_training']['save_frequency']
            )
            
            # Save final model
            agent_path = self.results_dir / f"dqn_agent_{participant_id}.pth"
            agent.save_model(str(agent_path))
            
            # Store agent and training history
            self.agents[participant_id] = agent
            self.training_history['rl_agents'][participant_id] = {
                'episode_rewards': agent.episode_rewards,
                'episode_lengths': agent.episode_lengths,
                'losses': agent.losses
            }
        
        logger.info("Stage 4 completed. All DQN agents trained.")
    
    def stage5_evaluation(self):
        """Stage 5: Comprehensive Model Evaluation"""
        logger.info("=== Stage 5: Model Evaluation ===")
        
        evaluation_results = {}
        
        # Evaluate each agent
        for participant_id, agent in self.agents.items():
            logger.info(f"Evaluating agent for participant {participant_id}")
            
            env = self.environments[participant_id]
            
            # Standard evaluation
            eval_results = agent.evaluate(
                env=env,
                episodes=self.config['evaluation']['test_episodes']
            )
            
            # Additional analysis
            test_states = [env.reset() for _ in range(20)]
            action_preferences = agent.get_action_preferences(test_states)
            
            # Stress reduction analysis
            stress_analysis = self._analyze_stress_reduction(agent, env)
            
            # Compile results
            evaluation_results[participant_id] = {
                'standard_metrics': eval_results,
                'action_preferences': action_preferences,
                'stress_analysis': stress_analysis,
                'scent_preferences': env.scent_preferences,
                'safety_compliance': self._evaluate_safety_compliance(agent, env)
            }
            
            # Save individual results
            result_path = self.results_dir / f"evaluation_{participant_id}.json"
            with open(result_path, 'w') as f:
                json.dump(evaluation_results[participant_id], f, 
                         indent=2, default=str)
        
        # Generate comparative analysis
        comparative_analysis = self._generate_comparative_analysis(evaluation_results)
        
        # Save comprehensive results
        all_results = {
            'individual_results': evaluation_results,
            'comparative_analysis': comparative_analysis,
            'training_history': self.training_history,
            'configuration': self.config
        }
        
        final_results_path = self.results_dir / 'final_evaluation_results.json'
        with open(final_results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Generate evaluation plots
        self._plot_evaluation_results(evaluation_results)
        
        logger.info("Stage 5 completed. Comprehensive evaluation finished.")
        
        return evaluation_results
    
    def _analyze_stress_reduction(self, agent: DQNAgent, env: ScentTherapyEnvironment) -> Dict:
        """Analyze stress reduction effectiveness"""
        stress_reductions = []
        scent_effectiveness = {}
        
        for _ in range(20):  # Multiple test episodes
            state = env.reset()
            initial_stress = env._calculate_stress_index()
            
            episode_scents = []
            for step in range(50):
                action = agent.act(state, training=False)
                next_state, reward, done, info = env.step(action)
                
                if info['scent_action'].scent_type != 'none':
                    episode_scents.append(info['scent_action'].scent_type)
                
                state = next_state
                if done:
                    break
            
            final_stress = env._calculate_stress_index()
            stress_reduction = initial_stress - final_stress
            stress_reductions.append(stress_reduction)
            
            # Track scent effectiveness
            for scent in set(episode_scents):
                if scent not in scent_effectiveness:
                    scent_effectiveness[scent] = []
                scent_effectiveness[scent].append(stress_reduction)
        
        # Calculate averages
        avg_effectiveness = {}
        for scent, reductions in scent_effectiveness.items():
            avg_effectiveness[scent] = {
                'mean_stress_reduction': float(np.mean(reductions)),
                'std_stress_reduction': float(np.std(reductions)),
                'usage_count': len(reductions)
            }
        
        return {
            'average_stress_reduction': float(np.mean(stress_reductions)),
            'std_stress_reduction': float(np.std(stress_reductions)),
            'scent_effectiveness': avg_effectiveness
        }
    
    def _evaluate_safety_compliance(self, agent: DQNAgent, env: ScentTherapyEnvironment) -> Dict:
        """Evaluate safety compliance of the agent"""
        safety_violations = 0
        total_actions = 0
        overuse_incidents = 0
        
        for _ in range(10):  # Test episodes
            state = env.reset()
            hourly_count = 0
            
            for step in range(100):
                action = agent.act(state, training=False)
                next_state, reward, done, info = env.step(action)
                
                total_actions += 1
                
                # Check for safety violations
                if info['safety_violation']:
                    safety_violations += 1
                
                # Check for overuse
                if info['scent_action'].scent_type != 'none':
                    hourly_count += 1
                    if hourly_count > 6:  # More than 6 scents per hour
                        overuse_incidents += 1
                
                state = next_state
                if done:
                    break
        
        return {
            'safety_violation_rate': safety_violations / max(total_actions, 1),
            'overuse_rate': overuse_incidents / max(total_actions, 1),
            'total_actions_evaluated': total_actions
        }
    
    def _generate_comparative_analysis(self, evaluation_results: Dict) -> Dict:
        """Generate comparative analysis across all participants"""
        
        # Collect metrics across participants
        metrics = {
            'average_rewards': [],
            'stress_reductions': [],
            'scent_usage_efficiency': [],
            'safety_scores': []
        }
        
        participant_rankings = {}
        
        for participant_id, results in evaluation_results.items():
            metrics['average_rewards'].append(results['standard_metrics']['average_reward'])
            metrics['stress_reductions'].append(results['standard_metrics']['average_stress_reduction'])
            
            # Calculate efficiency (stress reduction per scent used)
            total_scents = sum(results['standard_metrics']['scent_usage'].values())
            efficiency = results['standard_metrics']['average_stress_reduction'] / max(total_scents, 1)
            metrics['scent_usage_efficiency'].append(efficiency)
            
            # Safety score
            safety_score = 1.0 - results['safety_compliance']['safety_violation_rate']
            metrics['safety_scores'].append(safety_score)
            
            # Calculate overall score
            overall_score = (
                results['standard_metrics']['average_reward'] * 0.3 +
                results['standard_metrics']['average_stress_reduction'] * 0.4 +
                efficiency * 0.2 +
                safety_score * 0.1
            )
            participant_rankings[participant_id] = overall_score
        
        # Generate statistics
        comparative_stats = {}
        for metric, values in metrics.items():
            comparative_stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        # Rank participants
        ranked_participants = sorted(participant_rankings.items(), 
                                   key=lambda x: x[1], reverse=True)
        
        return {
            'comparative_statistics': comparative_stats,
            'participant_rankings': dict(ranked_participants),
            'best_performing_participant': ranked_participants[0][0],
            'population_insights': self._extract_population_insights(evaluation_results)
        }
    
    def _extract_population_insights(self, evaluation_results: Dict) -> Dict:
        """Extract insights from population-level analysis"""
        
        # Most effective scents across population
        scent_effectiveness = {}
        for results in evaluation_results.values():
            for scent, metrics in results['stress_analysis']['scent_effectiveness'].items():
                if scent not in scent_effectiveness:
                    scent_effectiveness[scent] = []
                scent_effectiveness[scent].append(metrics['mean_stress_reduction'])
        
        avg_scent_effectiveness = {
            scent: float(np.mean(values))
            for scent, values in scent_effectiveness.items()
        }
        
        # Most preferred scents
        scent_preferences = {}
        for results in evaluation_results.values():
            for scent, count in results['standard_metrics']['scent_usage'].items():
                if scent not in scent_preferences:
                    scent_preferences[scent] = 0
                scent_preferences[scent] += count
        
        return {
            'most_effective_scents': dict(sorted(avg_scent_effectiveness.items(), 
                                                key=lambda x: x[1], reverse=True)),
            'most_used_scents': dict(sorted(scent_preferences.items(), 
                                          key=lambda x: x[1], reverse=True)),
            'population_size': len(evaluation_results),
            'average_treatment_success_rate': float(np.mean([
                1.0 if results['standard_metrics']['average_stress_reduction'] > 0.1 else 0.0
                for results in evaluation_results.values()
            ]))
        }
    
    def _plot_evaluation_results(self, evaluation_results: Dict):
        """Generate comprehensive evaluation plots"""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Reward comparison
        ax1 = plt.subplot(3, 3, 1)
        participants = list(evaluation_results.keys())
        rewards = [results['standard_metrics']['average_reward'] 
                  for results in evaluation_results.values()]
        
        bars = ax1.bar(participants, rewards)
        ax1.set_title('Average Reward by Participant')
        ax1.set_ylabel('Average Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, reward in zip(bars, rewards):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{reward:.1f}', ha='center', va='bottom')
        
        # 2. Stress reduction comparison
        ax2 = plt.subplot(3, 3, 2)
        stress_reductions = [results['standard_metrics']['average_stress_reduction']
                           for results in evaluation_results.values()]
        
        bars = ax2.bar(participants, stress_reductions, color='lightcoral')
        ax2.set_title('Average Stress Reduction by Participant')
        ax2.set_ylabel('Stress Reduction')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, reduction in zip(bars, stress_reductions):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{reduction:.2f}', ha='center', va='bottom')
        
        # 3. Scent usage heatmap
        ax3 = plt.subplot(3, 3, 3)
        scent_data = []
        scent_types = ['lavender', 'citrus', 'mint', 'eucalyptus', 'none']
        
        for participant in participants:
            participant_usage = []
            for scent in scent_types:
                usage = evaluation_results[participant]['standard_metrics']['scent_usage'].get(scent, 0)
                participant_usage.append(usage)
            scent_data.append(participant_usage)
        
        sns.heatmap(scent_data, xticklabels=scent_types, yticklabels=participants,
                   annot=True, fmt='d', cmap='YlOrRd', ax=ax3)
        ax3.set_title('Scent Usage Heatmap')
        
        # 4. Training curves for RL agents
        ax4 = plt.subplot(3, 3, 4)
        for participant_id in participants[:3]:  # Show first 3 to avoid clutter
            if participant_id in self.training_history['rl_agents']:
                rewards = self.training_history['rl_agents'][participant_id]['episode_rewards']
                # Smooth the curve
                smoothed_rewards = pd.Series(rewards).rolling(window=50).mean()
                ax4.plot(smoothed_rewards, label=participant_id, alpha=0.7)
        
        ax4.set_title('RL Training Progress')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Smoothed Reward')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Safety compliance
        ax5 = plt.subplot(3, 3, 5)
        safety_scores = [1.0 - results['safety_compliance']['safety_violation_rate']
                        for results in evaluation_results.values()]
        
        bars = ax5.bar(participants, safety_scores, color='lightgreen')
        ax5.set_title('Safety Compliance Score')
        ax5.set_ylabel('Safety Score (0-1)')
        ax5.tick_params(axis='x', rotation=45)
        ax5.set_ylim(0, 1.1)
        
        for bar, score in zip(bars, safety_scores):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.2f}', ha='center', va='bottom')
        
        # 6. Scent effectiveness across population
        ax6 = plt.subplot(3, 3, 6)
        all_scent_effectiveness = {}
        
        for results in evaluation_results.values():
            for scent, metrics in results['stress_analysis']['scent_effectiveness'].items():
                if scent not in all_scent_effectiveness:
                    all_scent_effectiveness[scent] = []
                all_scent_effectiveness[scent].append(metrics['mean_stress_reduction'])
        
        scent_names = list(all_scent_effectiveness.keys())
        effectiveness_data = [all_scent_effectiveness[scent] for scent in scent_names]
        
        ax6.boxplot(effectiveness_data, labels=scent_names)
        ax6.set_title('Scent Effectiveness Distribution')
        ax6.set_ylabel('Stress Reduction')
        ax6.tick_params(axis='x', rotation=45)
        
        # 7. Individual participant performance radar
        ax7 = plt.subplot(3, 3, 7, projection='polar')
        
        # Select best performing participant
        best_participant = max(participants, 
                             key=lambda p: evaluation_results[p]['standard_metrics']['average_reward'])
        
        metrics = ['Reward', 'Stress Reduction', 'Safety', 'Efficiency']
        best_results = evaluation_results[best_participant]
        
        values = [
            best_results['standard_metrics']['average_reward'] / 10,  # Normalize
            best_results['standard_metrics']['average_stress_reduction'] * 10,  # Scale up
            1.0 - best_results['safety_compliance']['safety_violation_rate'],
            best_results['stress_analysis']['average_stress_reduction'] * 5  # Scale up
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax7.plot(angles, values, 'o-', linewidth=2, label=f'Best: {best_participant}')
        ax7.fill(angles, values, alpha=0.25)
        ax7.set_xticks(angles[:-1])
        ax7.set_xticklabels(metrics)
        ax7.set_title('Best Participant Performance')
        
        # 8. Training loss for emotion model
        ax8 = plt.subplot(3, 3, 8)
        if 'emotion_model' in self.training_history:
            train_loss = self.training_history['emotion_model'].get('train_loss', [])
            val_loss = self.training_history['emotion_model'].get('val_loss', [])
            
            if train_loss and val_loss:
                ax8.plot(train_loss, label='Training Loss', alpha=0.7)
                ax8.plot(val_loss, label='Validation Loss', alpha=0.7)
                ax8.set_title('Emotion Model Training')
                ax8.set_xlabel('Epoch')
                ax8.set_ylabel('Loss')
                ax8.legend()
                ax8.grid(True, alpha=0.3)
        
        # 9. Overall system performance summary
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Calculate summary statistics
        avg_reward = np.mean(rewards)
        avg_stress_reduction = np.mean(stress_reductions) 
        avg_safety = np.mean(safety_scores)
        total_participants = len(participants)
        
        summary_text = f"""
        SYSTEM PERFORMANCE SUMMARY
        
        Participants Trained: {total_participants}
        Average Reward: {avg_reward:.2f}
        Average Stress Reduction: {avg_stress_reduction:.3f}
        Average Safety Score: {avg_safety:.3f}
        
        Best Participant: {best_participant}
        
        Training Completed: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        """
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'comprehensive_evaluation.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Evaluation plots saved")
    
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("Starting full training pipeline")
        start_time = datetime.now()
        
        try:
            # Stage 1: Data Collection
            training_data = self.stage1_data_collection()
            
            # Stage 2: Emotion Model Training
            self.stage2_emotion_model_training(training_data)
            
            # Stage 3: RL Setup
            self.stage3_rl_setup()
            
            # Stage 4: RL Training
            self.stage4_rl_training()
            
            # Stage 5: Evaluation
            evaluation_results = self.stage5_evaluation()
            
            # Pipeline summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            summary = {
                'pipeline_duration': str(duration),
                'participants_trained': len(self.config['data']['participants']),
                'emotion_model_accuracy': 'Check emotion_model_training.png',
                'average_rl_performance': {
                    'mean_reward': float(np.mean([
                        results['standard_metrics']['average_reward']
                        for results in evaluation_results.values()
                    ])),
                    'mean_stress_reduction': float(np.mean([
                        results['standard_metrics']['average_stress_reduction']
                        for results in evaluation_results.values()
                    ]))
                },
                'completion_time': end_time.isoformat()
            }
            
            # Save pipeline summary
            with open(self.results_dir / 'pipeline_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("Full pipeline completed successfully!")
            logger.info(f"Duration: {duration}")
            logger.info(f"Results saved in: {self.results_dir}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Elder Well-being AI Training Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--stage', type=str, choices=['1', '2', '3', '4', '5', 'full'],
                       default='full', help='Pipeline stage to run')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TrainingPipeline(config_path=args.config)
    
    # Run specified stage
    if args.stage == 'full':
        pipeline.run_full_pipeline()
    elif args.stage == '1':
        pipeline.stage1_data_collection()
    elif args.stage == '2':
        pipeline.stage2_emotion_model_training()
    elif args.stage == '3':
        pipeline.stage3_rl_setup()
    elif args.stage == '4':
        pipeline.stage4_rl_training()
    elif args.stage == '5':
        pipeline.stage5_evaluation()
    
    logger.info("Pipeline execution completed.")


if __name__ == "__main__":
    main()