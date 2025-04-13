# ppo_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self, input_dim=107, hidden_dim=256, output_dim=5):
        super(PolicyNet, self).__init__()
        
        # Main network layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Output layers
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
            
    def forward(self, x):
        # Ensure input is properly shaped
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.layers(x)
        action_probs = self.action_head(features)
        return action_probs

class ValueNet(nn.Module):
    def __init__(self, input_dim=107, hidden_dim=256):
        super(ValueNet, self).__init__()
        
        # Main network layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Output layer
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1)
            nn.init.constant_(module.bias, 0)
            
    def forward(self, x):
        # Ensure input is properly shaped
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.layers(x)
        value = self.value_head(features)
        return value

class PPOAgent:
    def __init__(self, input_dim=107, output_dim=5, hidden_dim=256, load_existing=False):
        self.policy_net = PolicyNet(input_dim, hidden_dim, output_dim)
        self.value_net = ValueNet(input_dim, hidden_dim)
        self.optimizer = optim.Adam([
            {'params': self.policy_net.parameters(), 'lr': 3e-4},
            {'params': self.value_net.parameters(), 'lr': 1e-3}
        ])
        
        # Memory management
        self.max_memory_size = 10000  # Increased memory size
        self.batch_size = 256  # Increased batch size
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': [],
            'values': [],
            'advantages': []
        }
        
        # Training parameters
        self.entropy_coef = 0.2  # Higher initial entropy
        self.min_entropy = 0.01  # Minimum entropy threshold
        self.entropy_annealing_steps = 200000  # Slower entropy decay
        self.entropy_recovery_rate = 0.0001  # Rate at which entropy recovers
        self.entropy_target = 0.05  # Target entropy level
        self.clip_param = 0.2  # PPO clip parameter
        self.value_clip = 0.5  # Value function clip parameter
        self.gamma = 0.99  # Discount factor
        self.lambda_ = 0.95  # GAE lambda
        self.epochs = 10  # Number of training epochs per update
        
        # Statistics
        self.total_reward = 0
        self.total_hands = 0
        self.wins = 0
        self.hands_played = 0
        
        # Opponent modeling
        self.opponent_history = {
            'actions': [],
            'positions': [],
            'stack_sizes': [],
            'betting_patterns': []
        }
        self.opponent_model = None  # Will be initialized in update_opponent_model
        
        # Load existing model if requested
        if load_existing:
            self.load_model()

    def select_action(self, state):
        """Select an action given the current state"""
        # Ensure state is a tensor and has the correct shape
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        with torch.no_grad():
            # Get action probabilities
            logits = self.policy_net(state)
            dist = torch.distributions.Categorical(logits=logits)
            
            # Sample action
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Get value estimate
            value = self.value_net(state)
            
            return action.item(), log_prob.item(), value.item()

    def store_transition(self, obs, action, reward, done):
        """Store a transition in the replay buffer"""
        # Convert observation to tensor if it's not already
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            
        # Convert action to tensor and ensure it's a single value
        if isinstance(action, tuple):
            action = action[0]  # Take the first element if it's a tuple
        action = torch.LongTensor([action])
        
        # Get action probabilities and log probs
        with torch.no_grad():
            logits = self.policy_net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(action.squeeze())
            
            # Get value estimate
            value = self.value_net(obs).squeeze()
        
        # Store the transition
        self.memory['states'].append(obs.detach())
        self.memory['actions'].append(action.detach())
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value.detach())
        self.memory['log_probs'].append(log_prob.detach())
        self.memory['dones'].append(float(done))

    def store_reward(self, reward, done, info=None):
        """Store a reward in memory"""
        if len(self.memory['rewards']) > 0:
            self.memory['rewards'][-1] = reward
            
            # Only count wins when the hand is actually won
            if done and info and info.get('hand_result') == 'win':
                self.wins += 1
                self.total_hands += 1
            elif done:
                self.total_hands += 1
                
            # Update total reward
            self.total_reward += reward

    def update_entropy_coef(self):
        """Update entropy coefficient based on training progress"""
        progress = min(self.hands_played / self.entropy_annealing_steps, 1.0)
        self.entropy_coef = self.entropy_coef * (1 - progress) + self.entropy_target * progress
        
        # Check current entropy and adjust if too low
        if len(self.memory['states']) > 0:
            with torch.no_grad():
                states = torch.cat(self.memory['states'])
                logits = self.policy_net(states)
                dist = torch.distributions.Categorical(logits=logits)
                current_entropy = dist.entropy().mean().item()
                
                if current_entropy < self.min_entropy:
                    # Increase entropy coefficient if entropy is too low
                    self.entropy_coef = min(self.entropy_coef * 1.5, self.entropy_target)

    def update(self, final_value=0):
        """Update the policy and value networks"""
        if len(self.memory['states']) < self.batch_size:
            return
            
        # Get a batch of transitions
        batch = self.get_batch()
        if batch is None:
            return
            
        # Compute returns and advantages
        returns = []
        advantages = []
        R = final_value
        
        for r, v, mask in zip(reversed(batch['rewards']), reversed(batch['values']), reversed(batch['dones'])):
            R = r + self.gamma * R * mask
            advantage = R - v
            returns.insert(0, R)
            advantages.insert(0, advantage)
            
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for epoch in range(self.epochs):
            # Get current policy distributions and values
            logits = self.policy_net(batch['states'])
            dist = torch.distributions.Categorical(logits=logits)
            current_values = self.value_net(batch['states']).squeeze()
            
            # Get new log probabilities
            new_log_probs = dist.log_prob(batch['actions'].squeeze())
            
            # Compute ratio and clipped ratio
            ratio = torch.exp(new_log_probs - batch['log_probs'])
            clipped_ratio = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param)
            
            # Compute losses with proper normalization
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # Ensure current_values and returns have the same shape
            if current_values.dim() == 0:
                current_values = current_values.unsqueeze(0)
            if returns.dim() == 0:
                returns = returns.unsqueeze(0)
            
            # Value loss with proper clipping
            value_loss = F.mse_loss(current_values, returns)
            value_loss = torch.clamp(value_loss, max=self.value_clip)
            
            # Compute entropy
            entropy = -dist.entropy().mean()
            
            # Total loss with entropy bonus
            policy_loss_total = policy_loss - self.entropy_coef * entropy
            value_loss_total = 0.5 * value_loss
            
            # Optimize policy with gradient clipping
            self.optimizer.zero_grad()
            policy_loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            # Optimize value with gradient clipping
            self.optimizer.zero_grad()
            value_loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            # Update entropy coefficient
            self.hands_played += 1
            if self.hands_played % self.entropy_annealing_steps == 0:
                self.entropy_coef = max(0.1, self.entropy_coef * 0.995)
        
        # Only clear memory if it's too large
        if len(self.memory['states']) > self.max_memory_size:
            self.memory = {k: [] for k in self.memory}

    def save_model(self, filename='poker_ppo.pth'):
        """Save the model to a file"""
        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'entropy_coef': self.entropy_coef,
            'hands_played': self.hands_played,
            'total_reward': self.total_reward,
            'total_hands': self.total_hands,
            'wins': self.wins
        }
        torch.save(checkpoint, filename)
        
    def load_model(self, filename='poker_ppo.pth'):
        """Load the model from a file if it exists"""
        try:
            checkpoint = torch.load(filename)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.value_net.load_state_dict(checkpoint['value_state_dict'])
            
            # Load optimizer state if it exists
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load entropy parameters if they exist
            if 'entropy_coef' in checkpoint:
                self.entropy_coef = checkpoint['entropy_coef']
            if 'hands_played' in checkpoint:
                self.hands_played = checkpoint['hands_played']
            if 'total_reward' in checkpoint:
                self.total_reward = checkpoint['total_reward']
            if 'total_hands' in checkpoint:
                self.total_hands = checkpoint['total_hands']
            if 'wins' in checkpoint:
                self.wins = checkpoint['wins']
        except FileNotFoundError:
            pass  # Silently continue if no model exists

    def get_batch(self):
        """Get a batch of transitions from memory"""
        if len(self.memory['states']) < self.batch_size:
            return None
            
        # Get indices for batch
        indices = np.random.choice(len(self.memory['states']), self.batch_size, replace=False)
        
        # Get batch data
        batch = {
            'states': torch.stack([self.memory['states'][i] for i in indices]),
            'actions': torch.stack([self.memory['actions'][i] for i in indices]),
            'rewards': torch.tensor([self.memory['rewards'][i] for i in indices], dtype=torch.float32),
            'values': torch.stack([self.memory['values'][i] for i in indices]),
            'log_probs': torch.stack([self.memory['log_probs'][i] for i in indices]),
            'dones': torch.tensor([self.memory['dones'][i] for i in indices], dtype=torch.float32)
        }
        
        return batch

    def update_opponent_model(self, opponent_action, position, stack_size, betting_pattern):
        """Update opponent model with new observations"""
        self.opponent_history['actions'].append(opponent_action)
        self.opponent_history['positions'].append(position)
        self.opponent_history['stack_sizes'].append(stack_size)
        self.opponent_history['betting_patterns'].append(betting_pattern)
        
        # Keep only recent history
        max_history = 1000
        for key in self.opponent_history:
            if len(self.opponent_history[key]) > max_history:
                self.opponent_history[key] = self.opponent_history[key][-max_history:]
                
    def get_opponent_features(self):
        """Extract opponent features from history"""
        if not self.opponent_history['actions']:
            return np.zeros(10)  # Default features if no history
            
        # Calculate basic statistics
        action_counts = np.bincount(self.opponent_history['actions'], minlength=5)
        position_avg = np.mean(self.opponent_history['positions'])
        stack_avg = np.mean(self.opponent_history['stack_sizes'])
        betting_avg = np.mean(self.opponent_history['betting_patterns'])
        
        # Calculate more advanced features
        action_entropy = -np.sum(action_counts * np.log(action_counts + 1e-10))
        position_std = np.std(self.opponent_history['positions'])
        stack_std = np.std(self.opponent_history['stack_sizes'])
        betting_std = np.std(self.opponent_history['betting_patterns'])
        
        # Combine features
        features = np.concatenate([
            action_counts / len(self.opponent_history['actions']),  # Normalized action frequencies
            [action_entropy, position_avg, position_std, stack_avg, stack_std, betting_avg, betting_std]
        ])
        
        return features
