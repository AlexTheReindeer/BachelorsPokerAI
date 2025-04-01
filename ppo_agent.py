# ppo_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.pi = nn.Linear(hidden_dim, act_dim)  # policy head
        self.v  = nn.Linear(hidden_dim, 1)        # value head

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.pi(x)
        value = self.v(x)
        return logits, value

class PPOAgent:
    """
    A minimal PPO agent for poker. 
    We'll store transitions after each action, then do an update at the end of each hand.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=128, lr=1e-3, gamma=0.99, clip_ratio=0.2, update_steps=5, batch_size=32):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.update_steps = update_steps
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNet(obs_dim, hidden_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # On-policy buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.done_flags = []

    def select_action(self, obs_np):
        """
        Returns an action ID (0=fold,1=call,2=raise, etc.).
        We'll adapt these IDs in ppo_player.py to actual betting amounts.
        """
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        logits, value = self.policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        self.states.append(obs_t)
        self.actions.append(action)
        self.log_probs.append(dist.log_prob(action))
        self.values.append(value.squeeze(0))
        
        # Store the action for the PPO player to use
        self.last_action = action.item()
        
        return self.last_action

    def store_reward(self, reward, done):
        """
        Called after each action or at end of hand. 
        If partial step, reward=0; if final, reward = net gain/loss.
        """
        if done:
            # If it's the final reward and we have actions, update the last reward
            if self.actions:
                self.rewards[-1] = reward
                self.done_flags[-1] = float(done)
            # If we have no actions but get a final reward, just store it
            else:
                self.rewards.append(reward)
                self.done_flags.append(float(done))
        else:
            # For intermediate steps, only store if we have a corresponding action
            if len(self.rewards) < len(self.actions):
                self.rewards.append(reward)
                self.done_flags.append(float(done))

    def update(self):
        """
        Update the policy and value networks using PPO
        """
        if len(self.actions) < self.batch_size:
            return

        # Convert stored data to tensors
        states = torch.cat(self.states, dim=0)  # Concatenate all state tensors
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        dones = torch.FloatTensor(self.done_flags)
        old_log_probs = torch.FloatTensor(self.log_probs)

        # Get current policy and value predictions
        action_probs, values = self.policy(states)
        dist = torch.distributions.Categorical(logits=action_probs)
        new_log_probs = dist.log_prob(actions)

        # Compute returns and advantages
        returns = []
        R = 0
        for r, done in zip(reversed(self.rewards), reversed(self.done_flags)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Ensure all tensors are on the same device and have matching sizes
        values = values.squeeze(-1)  # Remove last dimension
        values = values[:len(self.rewards)]  # Only use values for which we have rewards
        advantages = returns - values

        # Ensure all tensors have matching sizes
        min_size = min(len(self.actions), len(self.rewards))
        actions = actions[:min_size]
        new_log_probs = new_log_probs[:min_size]
        old_log_probs = old_log_probs[:min_size]
        advantages = advantages[:min_size]

        # Compute PPO loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)

        # Combined loss
        total_loss = policy_loss + 0.5 * value_loss

        # Single backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Clear memory
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.done_flags.clear()

    def save_model(self, filename):
        """Save the model to a file"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filename)
        
    def load_model(self, filename):
        """Load the model from a file"""
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
