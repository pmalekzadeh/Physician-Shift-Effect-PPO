import simpy
import numpy as np
import pandas as pd
from MM3_tandem import TandemQueue
from run_config import SIM_CONFIG, TRAIN_CONFIG, NETWORK_CONFIG, REPLAY_BUFFER_CONFIG
from replay_buffer import ReplayBuffer
import pickle
import os

def run_single_episode(sim_time):
    """
    Run a single simulation episode - stop at sim_time, only let busy servers finish current patient
    
    Args:
        sim_time (float): Duration of active arrival phase
        
    Returns:
        TandemQueue: Completed simulation system
    """
    env = simpy.Environment()
    system = TandemQueue(env, is_training=False)
    
    # First phase: Run until sim_time (accepting new arrivals)
    env.run(until=sim_time)
    print(f"\nReached time {sim_time}. Stopping new arrivals...")
    
    # Stop new arrivals by setting arrival rate to 0
    system.arrival_rate = 0
    
    # Second phase: Only let busy servers finish their current patient
    # Don't process any waiting customers in Q1 or Q2
    busy_servers = [server for server in system.servers if server.busy]
    
    if busy_servers:
        print(f"Letting {len(busy_servers)} busy servers finish current patients...")
        # Continue simulation only until busy servers finish
        while any(server.busy for server in busy_servers):
            env.run(env.now + 1)
        print(f"All busy servers finished at time {env.now}")
    else:
        print("No busy servers - episode ends immediately")
    
    # Report final state
    remaining_q1 = len(system.queue1)
    remaining_q2 = sum(len(server.queue2) for server in system.servers)
    print(f"Episode ended with {remaining_q1} customers in Q1, {remaining_q2} customers in Q2")
    
    return system

def pretrain_replay_buffer(num_episodes=None, sim_time=None):
    """
    Pre-fill replay buffer with random experiences using two-phase simulation
    
    Args:
        num_episodes (int): Number of episodes to run
        sim_time (float): Duration of arrival phase for each episode
    
    Returns:
        ReplayBuffer: Filled replay buffer
    """
    # Use configuration values if not specified
    if num_episodes is None:
        num_episodes = TRAIN_CONFIG.get('num_episodes', TRAIN_CONFIG['num_episodes'])
    if sim_time is None:
        sim_time = SIM_CONFIG['sim_time']
    
    print(f"\nStarting replay buffer pre-training:")
    print(f"Number of episodes: {num_episodes}")
    print(f"Simulation time per episode: {sim_time}")
    
    # Initialize replay buffer
    state_dim = REPLAY_BUFFER_CONFIG['state_dim']
    replay_buffer = ReplayBuffer(
        capacity=REPLAY_BUFFER_CONFIG['capacity'],
        state_dim=state_dim
    )
    
    total_experiences = 0
    action_counts = {0: 0, 1: 0}  # Track counts of each action
    
    for episode in range(num_episodes):
        print(f"\nStarting episode {episode + 1}/{num_episodes}")
        
        # Run complete episode
        system = run_single_episode(sim_time)
        
        # Process experiences from this episode
        system.process_decision_events()
        
        # Only save CSV files for the first 10 episodes
        if episode < 10:
            system.save_all_events(f"pretrain_episode_{episode + 1}.csv")
        # Add experiences to combined buffer and count actions
        episode_experiences = len(system.replay_buffer)
        for exp in system.replay_buffer.buffer:
            replay_buffer.add(*exp)
            action_counts[exp[1]] += 1  # Count the action (exp[1] is the action)
        
        total_experiences += episode_experiences
        
        # Print episode statistics
    
    
    print("\nPre-training complete!")
    print(f"Total experiences collected: {total_experiences}")
    print(f"Action distribution: Q1 (0): {action_counts[0]}, Q2i (1): {action_counts[1]}")
    print("Buffer statistics:", replay_buffer.get_statistics())
    
    return replay_buffer

def save_experiences(replay_buffer, filename="pretrained_experiences.csv"):
    os.makedirs("output", exist_ok=True)
    experiences = []
    for state, action, reward, next_state, start_time, done in replay_buffer.buffer:
        experiences.append({
            'state': state.tolist(),
            'action': int(action),
            'reward': float(reward),
            'next_state': (next_state.tolist() if next_state is not None else None),
            'start_time': float(start_time),
            'done': bool(done)
        })
    pd.DataFrame(experiences).to_csv(f'output/{filename}', index=False)
    print(f"\nSaved {len(experiences)} experiences to output/{filename}")


def save_replay_buffer(replay_buffer, filename="pretrain_replay_buffer.pkl"):
    """Save replay buffer to disk"""
    try:
        os.makedirs("output", exist_ok=True)
        filepath = os.path.join("output", filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(replay_buffer, f)
        print(f"\nReplay buffer saved to {filepath}")
        
    except Exception as e:
        print(f"Error saving replay buffer: {str(e)}")

def load_replay_buffer():
    """Load replay buffer from disk using pickle"""
    try:
        with open('output/replay_buffer.pkl', 'rb') as f:
            buffer = pickle.load(f)
            
            # Initialize with correct state dimension
            state_dim = REPLAY_BUFFER_CONFIG['state_dim']  # q1_length + q2_lengths + server_status
            replay_buffer = ReplayBuffer(
                capacity=REPLAY_BUFFER_CONFIG['capacity'],
                state_dim=state_dim
            )
            
            # Copy experiences from loaded buffer
            for exp in buffer.buffer:
                replay_buffer.add(*exp)
                
            print("\nLoaded replay buffer successfully")
            print("Buffer statistics:", replay_buffer.get_statistics())
            return replay_buffer
    except Exception as e:
        print(f"Error loading replay buffer: {e}")
        return None


if __name__ == "__main__":
    # Create and save replay buffer
    print("Creating replay buffer...")
    replay_buffer = pretrain_replay_buffer()
    
    # Save the buffer
    save_replay_buffer(replay_buffer)
    save_experiences(replay_buffer)