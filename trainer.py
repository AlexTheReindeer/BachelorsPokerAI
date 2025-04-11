# trainer.py

import sys
import random
import numpy as np
import pypokerengine
from pypokerengine.api.game import setup_config, start_poker
import torch
import argparse
import os

from ppo_agent import PPOAgent
from ppo_player import PPOPlayer, encode_observation
from random_player import RandomPlayer
from rule_based_player import RuleBasedPlayer

class PokerEnv:
    def __init__(self, opponent_type='self_play', load_existing=False):
        self.ppo_agent = PPOAgent(input_dim=117, output_dim=7, hidden_dim=256)
        self.ppo_player = PPOPlayer(self.ppo_agent)
        self.ppo_player.uuid = "Prodigy"  # Set unique identifier for PPO player
        
        # Set up opponent based on type
        self.opponent_type = opponent_type
        if opponent_type == 'self_play':
            # For self-play, use a new PPOPlayer instance for the opponent
            self.opponent = PPOPlayer(self.ppo_agent)
            self.opponent.uuid = "Trainer-Self"  # Unique identifier for self-play opponent
            opponent_name = "Trainer-Self"
        elif opponent_type == 'rule_based':
            self.opponent = RuleBasedPlayer()
            self.opponent.uuid = "Trainer-Rule"  # Unique identifier for rule-based opponent
            opponent_name = "Trainer-Rule"
        else:
            self.opponent = RandomPlayer()
            self.opponent.uuid = "Trainer-Random"  # Unique identifier for random opponent
            opponent_name = "Trainer-Random"
            
        self.config = setup_config(max_round=1, initial_stack=1000, small_blind_amount=10)
        self.config.register_player(name="Prodigy", algorithm=self.ppo_player)
        self.config.register_player(name=opponent_name, algorithm=self.opponent)
        
        # Store game result
        self.game_result = {'winners': []}
        
        # Mode will be set later
        self.mode = None

        if load_existing:
            self.load_model()

    def load_model(self, filename='poker_ppo.pth'):
        """Load the model from a file"""
        if os.path.exists(filename):
            print(f"Loading model from {filename}")
            self.ppo_agent.load_model(filename)
            return True
        else:
            print(f"No trained model found at {filename}, starting fresh")
            return False

    def save_model(self, filename='poker_ppo.pth'):
        """Save the model to a file"""
        print(f"Saving model to {filename}")
        self.ppo_agent.save_model(filename)

    def reset(self):
        """Reset the environment for a new episode"""
        # Start a new poker game
        game_state = start_poker(self.config, verbose=0)
        
        # Update game state for both players
        self.ppo_player.game_state = {
            'round_state': game_state,
            'hole_card': [],
            'community_card': [],
            'hand_over': False
        }
        
        if self.opponent_type != 'self_play':
            self.opponent.game_state = {
                'round_state': game_state,
                'hole_card': [],
                'community_card': [],
                'hand_over': False
            }
        
        # Reset player state
        self.ppo_player.current_reward = 0
        self.ppo_player.last_obs = None
        
        # Get initial observation
        return self.ppo_player.get_observation()

    def step(self, action):
        """Take an action in the environment"""
        try:
            # Store the action in the PPO agent
            self.ppo_agent.last_action = action
            
            # Start a new round with the current game state
            game_state = start_poker(self.config, verbose=0)
            
            # Update game states
            if game_state:
                self.ppo_player.game_state['round_state'] = game_state
                if self.opponent_type != 'self_play':
                    self.opponent.game_state['round_state'] = game_state
            
            # Get reward and next observation
            reward = self.ppo_player.get_reward()
            next_obs = self.ppo_player.get_observation()
            done = self.ppo_player.is_hand_over()
            
            # Store game result if hand is over
            info = {}
            if done:
                # Determine winner based on reward
                winner_uuid = "Prodigy" if reward > 0 else self.opponent.uuid
                self.game_result = {'winners': [{'uuid': winner_uuid}]}
                
                # Add hand result to info
                info['hand_result'] = 'win' if reward > 0 else 'loss'
                
                # Reset the environment for the next hand
                self.reset()
            
            return next_obs, reward, done, info
            
        except Exception as e:
            print(f"Error in step: {e}")
            return self.reset(), 0, True, {}

    def train(self, num_episodes):
        """Main training loop"""
        # Track wins and losses
        wins = 0
        losses = 0
        total_reward = 0
        best_win_rate = 0
        episode_rewards = []
        
        print("\nStarting training...")
        print(f"Initial win rate: {(self.ppo_agent.wins / max(1, self.ppo_agent.total_hands)) * 100:.2f}%")
        
        # Training loop
        for episode in range(num_episodes):
            obs = self.reset()
            done = False
            episode_reward = 0
            hand_reward = 0
            
            while not done:
                # Get opponent features
                opponent_features = self.ppo_agent.get_opponent_features()
                
                # Ensure observation is the correct size and type
                obs = np.array(obs, dtype=np.float32)
                if len(obs) < 107:  # Pad if needed
                    obs = np.pad(obs, (0, 107 - len(obs)))
                elif len(obs) > 107:  # Truncate if needed
                    obs = obs[:107]
                
                # Combine observation with opponent features
                obs_with_opponent = np.concatenate([obs, opponent_features])
                
                # Convert to tensor and ensure proper shape
                obs_tensor = torch.FloatTensor(obs_with_opponent)
                
                # Select and execute action
                action = self.ppo_agent.select_action(obs_tensor)
                next_obs, reward, done, info = self.step(action)
                
                # Update opponent model
                if info and 'opponent_action' in info:
                    self.ppo_agent.update_opponent_model(
                        info['opponent_action'],
                        info.get('opponent_position', 0),
                        info.get('opponent_stack', 0),
                        info.get('opponent_betting', 0)
                    )
                
                # Store transition
                self.ppo_agent.store_transition(obs_tensor, action, reward, done)
                obs = next_obs
                hand_reward += reward
                episode_reward += reward
                
                # Track wins and losses
                if done:
                    if info and info.get('hand_result') == 'win':
                        wins += 1
                    else:
                        losses += 1
            
            total_reward += episode_reward
            episode_rewards.append(episode_reward)
            
            # Update policy at the end of each episode
            self.ppo_agent.update()
            
            # Save model if win rate improves
            if (episode + 1) % 500 == 0:  # Check every 500 episodes
                current_win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
                if current_win_rate > best_win_rate:
                    best_win_rate = current_win_rate
                    self.ppo_agent.save_model()
            
            # Progress updates
            if (episode + 1) % 500 == 0:  # Show progress every 500 episodes
                win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
                avg_reward = np.mean(episode_rewards[-500:]) if episode_rewards else 0
                
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"Win Rate: {win_rate:.2%}")
                print(f"Best Win Rate: {best_win_rate:.2%}")
                print(f"Total Reward: {total_reward}")
                print(f"Average Reward (last 500): {avg_reward:.2f}")
                print(f"Entropy Coefficient: {self.ppo_agent.entropy_coef:.4f}")
                print("-" * 50)
        
        # Print final statistics
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        print("\nTraining Complete!")
        print(f"Total Episodes: {num_episodes}")
        print(f"Final Win Rate: {win_rate:.2%}")
        print(f"Best Win Rate: {best_win_rate:.2%}")
        print(f"Total Reward: {total_reward}")
        print(f"Average Reward per Episode: {total_reward/num_episodes:.2f}")

def train_ppo_poker(num_hands=1000):
    """Train the PPO agent through self-play"""
    # Create environment with self-play opponent
    env = PokerEnv(opponent_type='self_play', load_existing=True)
    
    # Train the agent
    env.train(num_hands)
    
    # Save the final model
    env.ppo_agent.save_model()

def test_ppo_poker(num_hands=1000):
    # Test against rule-based opponent
    print("\nTesting against rule-based opponent...")
    env = PokerEnv(opponent_type='rule_based', load_existing=True)
    env.mode = 'test'
    
    # Track wins and losses
    wins = 0
    losses = 0
    total_reward = 0
    
    # Testing loop
    for episode in range(num_hands):
        obs = env.reset()
        done = False
        episode_reward = 0
        hand_result = None  # Will be 'win', 'loss', or None
        
        while not done:
            action = env.ppo_agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            episode_reward += reward
            
            # Track hand result
            if done:
                if info.get('hand_result') == 'win':
                    wins += 1
                    hand_result = 'win'
                elif info.get('hand_result') == 'loss':
                    losses += 1
                    hand_result = 'loss'
        
        total_reward += episode_reward
        
        # Progress updates
        if (episode + 1) % 100 == 0:  # Show progress every 100 episodes
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            print(f"Episode {episode + 1}/{num_hands}")
            print(f"Hand Win Rate: {win_rate:.2%}")
            print(f"Total Reward: {total_reward}")
            print(f"Average Reward per Episode: {total_reward/(episode+1):.2f}")
            print("-" * 50)
    
    # Print final statistics
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    print("\nTesting Complete!")
    print(f"Total Episodes: {num_hands}")
    print(f"Hand Win Rate: {win_rate:.2%}")
    print(f"Total Reward: {total_reward}")
    print(f"Average Reward per Episode: {total_reward/num_hands:.2f}")

def compare_ppo_poker(num_hands=1000):
    # Test before training
    print("\nTesting against rule-based opponent BEFORE training...")
    env = PokerEnv(opponent_type='rule_based')
    env.mode = 'compare'
    
    # Load model if it exists
    env.load_model()
        
    env.train(num_hands)
    
    # Train with self-play
    print("\nStarting self-play training...")
    env = PokerEnv(opponent_type='self_play')
    env.mode = 'compare'
    
    # Always load the model after the first test phase
    env.load_model()
    
    env.train(num_hands)
    
    # Test after training
    print("\nTesting against rule-based opponent AFTER training...")
    env = PokerEnv(opponent_type='rule_based')
    env.mode = 'compare'
    
    # Always load the model after the training phase
    env.load_model()
    
    env.train(num_hands)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test a PPO poker agent')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'compare'],
                      help='Mode to run in: train, test, or compare')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='Number of episodes to run')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_ppo_poker(args.episodes)
    elif args.mode == 'test':
        test_ppo_poker(args.episodes)
    elif args.mode == 'compare':
        compare_ppo_poker(args.episodes)
