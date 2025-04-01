# trainer.py

import sys
import random
import numpy as np
import pypokerengine
from pypokerengine.api.game import setup_config, start_poker
import torch
import argparse

from ppo_agent import PPOAgent
from ppo_player import PPOPlayer, encode_observation
from random_player import RandomPlayer
from rule_based_player import RuleBasedPlayer

class PokerEnv:
    def __init__(self, opponent_type='self_play'):
        self.ppo_agent = PPOAgent(obs_dim=10, act_dim=3)
        
        # Always try to load the trained model
        try:
            self.ppo_agent.policy.load_state_dict(torch.load('poker_ppo.pth'))
            print("Loaded trained model from poker_ppo.pth")
        except:
            print("No trained model found at poker_ppo.pth, starting fresh")
            
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
        # Store the action in the PPO agent
        self.ppo_agent.last_action = action
        
        # Start a new round with the current game state
        game_state = start_poker(self.config, verbose=0)
        
        # Update game states
        self.ppo_player.game_state['round_state'] = game_state
        if self.opponent_type != 'self_play':
            self.opponent.game_state['round_state'] = game_state
        
        # Get reward and next observation
        reward = self.ppo_player.get_reward()
        next_obs = self.ppo_player.get_observation()
        done = self.ppo_player.is_hand_over()
        
        # Store game result if hand is over
        if done:
            # Determine winner based on reward
            winner_uuid = "Prodigy" if reward > 0 else self.opponent.uuid
            self.game_result = {'winners': [{'uuid': winner_uuid}]}
            
            # Reset the environment for the next hand
            self.reset()
        
        return next_obs, reward, done, {}

def train(env, episodes, save_interval=100):
    """Train the agent for a specified number of episodes"""
    wins = 0
    losses = 0
    total_reward = 0
    
    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = env.ppo_player.ppo_agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
        # Track wins/losses based on the winner's UUID
        if env.game_result and 'winners' in env.game_result and len(env.game_result['winners']) > 0:
            winner = env.game_result['winners'][0]
            # Check if winner's UUID is Prodigy
            if winner.get('uuid') == "Prodigy":
                wins += 1
            else:
                losses += 1
        
        total_reward += episode_reward
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
            print(f"Episode {episode + 1}/{episodes}")
            print(f"Win rate: {win_rate:.1f}% ({wins}/{wins + losses})")
            print(f"Total reward: {total_reward:.0f}")
            print("-------------------")
    
    # Save the final model at the end of training
    env.ppo_player.ppo_agent.save_model("poker_ppo.pth")
    print(f"\nTraining complete. Final model saved to 'poker_ppo.pth'")

def train_ppo_poker(num_hands=1000):
    # Build a PPOAgent with obs_dim=10, act_dim=3 (fold/call/raise)
    ppo_agent = PPOAgent(obs_dim=10, act_dim=3)

    # PPO seat
    seat0 = PPOPlayer(ppo_agent)
    # Opponent seat
    seat1 = RandomPlayer()

    config = setup_config(max_round=num_hands, initial_stack=1000, small_blind_amount=10)
    config.register_player(name="ppo_seat0", algorithm=seat0)
    config.register_player(name="random_seat1", algorithm=seat1)

    # Run the game
    game_result = start_poker(config, verbose=0)
    print("Game result:", game_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test poker AI')
    parser.add_argument('--mode', choices=['test', 'train', 'compare'], default='test',
                      help='Mode to run: test (against rule-based), train (self-play), or compare (test before and after training)')
    parser.add_argument('--episodes', type=int, default=100,
                      help='Number of episodes to run')
    args = parser.parse_args()
    
    if args.mode == 'test':
        # Test against rule-based opponent
        print("\nTesting against rule-based opponent...")
        env = PokerEnv(opponent_type='rule_based')
        train(env, args.episodes)
        
    elif args.mode == 'train':
        # Train with self-play
        print("\nStarting self-play training...")
        env = PokerEnv(opponent_type='self_play')
        train(env, args.episodes)
        
    elif args.mode == 'compare':
        # Test before training
        print("\nTesting against rule-based opponent BEFORE training...")
        env = PokerEnv(opponent_type='rule_based')
        train(env, args.episodes)
        
        # Train with self-play
        print("\nStarting self-play training...")
        env = PokerEnv(opponent_type='self_play')
        train(env, args.episodes)
        
        # Test after training
        print("\nTesting against rule-based opponent AFTER training...")
        env = PokerEnv(opponent_type='rule_based')
        train(env, args.episodes)
