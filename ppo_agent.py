import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from run_config import NETWORK_CONFIG, TRAIN_CONFIG

class PPONetwork(nn.Module):
    """Actor-Critic network for PPO"""
    def __init__(self, state_dim, action_dim, hidden_dims):
        super(PPONetwork, self).__init__()
        
        # Shared layers
        self.shared_layers = nn.ModuleList()
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.shared_layers.append(linear)
            prev_dim = hidden_dim
        
        # Actor head (policy)
        self.actor = nn.Linear(prev_dim, action_dim)
        nn.init.xavier_uniform_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)
        
        # Critic head (value function)
        self.critic = nn.Linear(prev_dim, 1)
        nn.init.xavier_uniform_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)
        
    def forward(self, x):
        # Shared feature extraction
        for layer in self.shared_layers:
            x = F.relu(layer(x))
        
        # Actor and critic outputs
        action_logits = self.actor(x)
        value = self.critic(x)
        
        return action_logits, value

class PPOBuffer:
    """Buffer for storing PPO experiences"""
    def __init__(self, capacity, state_dim, device):
        self.capacity = capacity
        self.device = device
        self.state_dim = state_dim
        
        # Storage
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.values = deque(maxlen=capacity)
        self.log_probs = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        self.advantages = deque(maxlen=capacity)
        self.returns = deque(maxlen=capacity)
        
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def compute_gae(self, next_value, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        # Work backwards through the buffer
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[i]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[i]
                next_val = self.values[i + 1]
            
            delta = self.rewards[i] + gamma * next_val * next_non_terminal - self.values[i]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages.insert(0, gae)
        
        # Compute returns
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        # Store advantages and returns
        self.advantages.extend(advantages)
        self.returns.extend(returns)
        
    def get_batches(self, batch_size):
        """Get batches for training"""
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(self.advantages).to(self.device)
        returns = torch.FloatTensor(self.returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Create batches
        dataset_size = len(states)
        indices = torch.randperm(dataset_size)
        
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            
            yield (
                states[batch_indices],
                actions[batch_indices],
                old_log_probs[batch_indices],
                advantages[batch_indices],
                returns[batch_indices]
            )
    
    def clear(self):
        """Clear the buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()
    
    def __len__(self):
        return len(self.states)

class PPOAgent:
    def __init__(self, state_dim, action_dim, device, **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Get network architecture from config
        self.hidden_dims = NETWORK_CONFIG['hidden_dims']
        
        # Initialize PPO hyperparameters
        self.lr = TRAIN_CONFIG.get('lr', 3e-4)
        self.gamma = TRAIN_CONFIG.get('gamma', 0.99)
        self.gae_lambda = TRAIN_CONFIG.get('gae_lambda', 0.95)
        self.clip_ratio = TRAIN_CONFIG.get('clip_ratio', 0.2)
        self.value_loss_coef = TRAIN_CONFIG.get('value_loss_coef', 0.5)
        self.entropy_coef = TRAIN_CONFIG.get('entropy_coef', 0.01)
        self.max_grad_norm = TRAIN_CONFIG.get('max_grad_norm', 0.5)
        self.ppo_epochs = TRAIN_CONFIG.get('ppo_epochs', 4)
        self.batch_size = TRAIN_CONFIG.get('batch_size', 64)
        self.buffer_size = TRAIN_CONFIG.get('buffer_size', 2048)
        
        # Initialize network
        self.network = PPONetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.hidden_dims
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
        # Initialize buffer
        self.buffer = PPOBuffer(
            capacity=self.buffer_size,
            state_dim=self.state_dim,
            device=self.device
        )
        
        # Training tracking
        self.steps_done = 0
        self.episode_count = 0
        
        # Override with any provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        print(f"PPO Agent initialized with:")
        print(f"State dim: {state_dim}")
        print(f"Action dim: {action_dim}")
        print(f"Hidden dims: {self.hidden_dims}")
        print(f"Device: {device}")
        print(f"Learning rate: {self.lr}")
        print(f"Buffer size: {self.buffer_size}")

    def select_action(self, state):
        """Select action using current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_logits, value = self.network(state_tensor)
            
            # Create action distribution
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            
            # Sample action
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()

    def get_value(self, state):
        """Get value estimate for a state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, value = self.network(state_tensor)
            return value.item()

    def add_experience(self, state, action, reward, value, log_prob, done):
        """Add experience to buffer"""
        self.buffer.add(state, action, reward, value, log_prob, done)

    def update(self):
        """Update the PPO agent"""
        if len(self.buffer) < self.batch_size:
            return None
        
        # Compute GAE and returns
        with torch.no_grad():
            # Get next value for GAE computation
            if len(self.buffer.states) > 0:
                last_state = self.buffer.states[-1]
                next_value = self.get_value(last_state)
            else:
                next_value = 0.0
            
            self.buffer.compute_gae(next_value, self.gamma, self.gae_lambda)
        
        # Training statistics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_loss = 0
        
        # PPO update epochs
        for epoch in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                states, actions, old_log_probs, advantages, returns = batch
                
                # Forward pass
                action_logits, values = self.network(states)
                action_probs = F.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                
                # Compute losses
                new_log_probs = action_dist.log_prob(actions)
                entropy = action_dist.entropy().mean()
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), returns)
                
                # Total loss
                loss = (policy_loss + 
                       self.value_loss_coef * value_loss - 
                       self.entropy_coef * entropy)
                
                # Check for invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected: {loss.item()}")
                    continue
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Accumulate statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
                total_loss += loss.item()
        
        # Clear buffer after update
        self.buffer.clear()
        
        # Return average losses
        num_updates = self.ppo_epochs * max(1, len(self.buffer.states) // self.batch_size)
        if num_updates > 0:
            return {
                'total_loss': total_loss / num_updates,
                'policy_loss': total_policy_loss / num_updates,
                'value_loss': total_value_loss / num_updates,
                'entropy_loss': total_entropy_loss / num_updates
            }
        else:
            return None

    def save(self, path):
        """Save the model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episode_count': self.episode_count
        }, path)

    def load(self, path):
        """Load the model"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)
        self.episode_count = checkpoint.get('episode_count', 0)

    def get_action_probabilities(self, state):
        """Get action probabilities for a state (for analysis)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_logits, _ = self.network(state_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
            return action_probs.cpu().numpy().flatten()
