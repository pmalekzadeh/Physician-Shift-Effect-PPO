import numpy as np
from collections import deque
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        """
        Initialize Replay Buffer
        
        Args:
            capacity (int): Maximum size of the buffer
            state_dim (int): Dimension of the state space
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.buffer = []
        self.position = 0
        
        # Track statistics
        self.insertion_count = 0
        self.sample_count = 0
    
    def add(self, state, action, reward, next_state, end_time, done):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # Create experience tuple
        experience = (
            np.array(state, dtype=np.float32),
            action,
            reward,
            np.array(next_state, dtype=np.float32) if next_state is not None else np.zeros(self.state_dim, dtype=np.float32),
            end_time,
            done
        )
        
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        self.insertion_count += 1

    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        if not self.can_sample(batch_size):
            raise ValueError(f"Not enough samples in buffer to fetch {batch_size} items. Current size: {len(self.buffer)}")
        
        batch = random.sample(self.buffer, batch_size)
        state_batch = np.array([experience[0] for experience in batch])
        action_batch = np.array([experience[1] for experience in batch])
        reward_batch = np.array([experience[2] for experience in batch])
        next_state_batch = np.array([experience[3] for experience in batch])
        end_time_batch = np.array([experience[4] for experience in batch])
        done_batch = np.array([experience[5] for experience in batch])
        
        return (state_batch, action_batch, reward_batch, next_state_batch, end_time_batch, done_batch)
    
    def can_sample(self, batch_size):
        """Check if enough experiences are available for sampling"""
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        
    def get_statistics(self):
        """Get buffer statistics"""
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'insertions': self.insertion_count,
            'samples': self.sample_count
        }
    
    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, state_dim, alpha=0.6, beta=0.4):
        """
        Initialize Prioritized Replay Buffer
        
        Args:
            capacity (int): Maximum size of the buffer
            state_dim (int): Dimension of the state space
            alpha (float): Priority exponent
            beta (float): Importance sampling exponent
        """
        super().__init__(capacity, state_dim)
        self.alpha = alpha
        self.beta = beta
        self.priorities = deque(maxlen=capacity)
        self.eps = 1e-6  # Small constant to prevent zero probabilities
        
    def add(self, state, action, reward, next_state, done, start_time):
        """Add experience with maximum priority"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)
        super().add(state, action, reward, next_state, done, start_time )
        
    def sample(self, batch_size):
        """Sample batch based on priorities"""
        if not self.can_sample(batch_size):
            raise ValueError("Not enough experiences in buffer")
            
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices and calculate importance weights
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        weights = torch.FloatTensor(weights)
        
        # Get experiences
        batch = super().sample(batch_size)
        return batch + (weights, indices)
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + self.eps