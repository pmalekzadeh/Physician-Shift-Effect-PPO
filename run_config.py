"""
Configuration file for the tandem queue simulation and RL training parameters.
"""

# Simulation Parameters
SIM_CONFIG = {
    'num_servers': 1,
    'arrival_rate': 4,
    'service_rate_1': 3,
    'service_rate_2': 3,
    'alpha': 0.7,  # Weight for T1 sojourn time in reward
    'beta': 0.8,   # Weight for T2 sojourn time in reward
    'theta': 3,  # Probability of patient abandonment from Q1 per time unit
    'base_seed': 42,   # Base random seed for reproducibility
    'sim_time': 50,  # Duration of each simulation episode,
    'reward_type': 'physician',
    'q1_capacity': 20,  # Maximum capacity for Q1 queue
    'q2_capacity': 20   # Maximum capacity for Q2 queue (default same as q1_capacity)
}

# RL Training Parameters
TRAIN_CONFIG = {
    'batch_size': 256,         # Reduced batch size for stability
    'lr': 3e-4,           # Much lower learning rate for stability
    'gamma': 0.99,            # Slightly lower discount factor
    'epsilon_start': 1,       # Starting exploration rateq1_capcity an
    'epsilon_end': 0.1,      # Final exploration rate
    'epsilon_decay': 0.999,   # Much slower decay for more exploration
    'target_update_steps': 1000,  # Less frequent target updates for stability
    'replay_buffer_size': 8000000,
    'random_seed': 42,        # Random seed for training
    'train_num_episodes': 5000,  # Number of episodes to iterate over the replay buffer
    'num_episodes': 5000,  # Number of episodes to pretrain the buffer

    # 'modified_states_type': 'Q1Q2s'
    # 'modified_states_type': 'Q1Q2sServerid'
    'modified_states_type': 'Q1Q2',
    'agent_type': 'ppo',
    
    # PPO-specific hyperparameters
    'gae_lambda': 0.95,       # GAE lambda parameter
    'clip_ratio': 0.2,        # PPO clipping parameter
    'value_loss_coef': 0.5,   # Value loss coefficient
    'entropy_coef': 0.01,     # Entropy bonus coefficient
    'max_grad_norm': 0.5,     # Gradient clipping norm
    'ppo_epochs': 4,          # Number of PPO update epochs
    'buffer_size': 2048       # PPO buffer size
}

# RL Ecluation Parameters
EVAL_CONFIG = {
    'num_episodes': 2,
    'num_runs': 1000}

# Replay Buffer Configuration
REPLAY_BUFFER_CONFIG = {
    'state_dim': 3,  # [q1_length, q2_server_length, time] for Q1Q2 mode
    'capacity': TRAIN_CONFIG['replay_buffer_size'],
    'min_memory_size': 100
}

# Network Architecture
NETWORK_CONFIG = {
    'action_dim': 2,        # Number of possible actions (Q1 or Q2)
    'hidden_dims': [128, 128] # Hidden layer dimensions (matches saved model)
}

def print_config():
    """Print the current configuration settings."""
    print("\nSimulation Configuration:")
    for key, value in SIM_CONFIG.items():
        print(f"{key}: {value}")
    
    print("\nTraining Configuration:")
    for key, value in TRAIN_CONFIG.items():
        print(f"{key}: {value}")
    
    print("\nNetwork Configuration:")
    for key, value in NETWORK_CONFIG.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    print_config() 