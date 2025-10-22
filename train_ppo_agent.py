#!/usr/bin/env python3
"""
Training script for PPO agent on the tandem queue simulation.
- Steps only at decision epochs (q1>0 & q2>0).
- Environment rolls through forced actions until the next decision epoch (or terminal).
- Reward passed to PPO is the cumulative reward between decision epochs (what env.step returns).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import simpy
from simpy.core import EmptySchedule

from MM3_tandem import TandemQueue
from ppo_agent import PPOAgent
from run_config import SIM_CONFIG, TRAIN_CONFIG, NETWORK_CONFIG, print_config
# Optional: only keep this if your loader is PPO-aware (on-policy)
# from pretrain_buffer import load_and_transfer_pretrained_data


# ---------------------------
# State mapping
# ---------------------------
def _map_state_for_training(state_vec):
    """
    Return (state_modified, q1_len, q2_len) according to TRAIN_CONFIG['modified_states_type'].
    - 'Q1Q2'   : state is [q1, q2_server, time]  -> return as-is
    - 'Q1Q2s'  : map to [q1, q2_current_server]
    - default  : full state [q1, q2_0, q2_1, ..., server_id, time]
    """
    mtype = TRAIN_CONFIG.get('modified_states_type', None)

    if mtype == 'Q1Q2':
        q1_len = int(state_vec[0])
        q2_len = int(state_vec[1])
        state_modified = state_vec
        return state_modified, q1_len, q2_len

    elif mtype == 'Q1Q2s':
        sid = int(state_vec[-2])              # server_id
        q1_len = int(state_vec[0])
        q2_len = int(state_vec[1 + sid])      # q2 length for that server
        state_modified = np.array([q1_len, q2_len, float(state_vec[-1])], dtype=np.float32)  # keep time as 3rd
        return state_modified, q1_len, q2_len

    else:
        # full state: [q1, q2_0, q2_1, ..., server_id, time]
        sid = int(state_vec[-2])
        q1_len = int(state_vec[0])
        q2_len = int(state_vec[1 + sid])
        state_modified = state_vec
        return state_modified, q1_len, q2_len


# ---------------------------
# PPO training loop (aligned with your DQN loop)
# ---------------------------
def train_ppo_agent(agent: PPOAgent):
    losses = []
    episode_rewards = []
    all_events = []       # per-episode decision events (annotated)
    env_all_events = []   # raw env event stream

    # for plotting breakdown (if agent.update returns stats)
    policy_losses, value_losses, entropy_losses = [], [], []

    for episode in range(TRAIN_CONFIG['train_num_episodes']):
        env = TandemQueue(simpy.Environment(), agent=agent)

        # ---------- Bootstrap to FIRST decision epoch (or terminal) ----------
        try:
            ticks = 0
            while (env.current_decision_state is None) and (not env.is_simulation_complete()):
                env.env.step()
                env.try_forced_service_on_idle_servers()
                ticks += 1
                if ticks > 1_000_000:
                    raise RuntimeError("Bootstrap stuck (no decision epoch reached).")
        except EmptySchedule:
            pass

        # If we never reached a decision epoch, skip this episode
        if env.current_decision_state is None or env.is_simulation_complete():
            episode_rewards.append(0.0)
            avg_loss = np.mean(losses[-10:]) if losses else 0.0
            print(f"Episode {episode + 1:3d}/{TRAIN_CONFIG['train_num_episodes']} | "
                  f"Reward: {0.0:8.2f} | Events: 0 | Avg Loss: {avg_loss:.4f} "
                  f"(no decision epoch)")
            continue

        # Capture the initial decision state
        state = env.get_state_array(env.current_decision_state)
        
        # Validate initial decision state
        state_modified, q1_len, q2_len = _map_state_for_training(state)
        if not (q1_len > 0 and q2_len > 0 and q2_len < SIM_CONFIG['q2_capacity']):
            raise ValueError(f"Invalid initial decision state: q1={q1_len}, q2={q2_len}, q2_capacity={SIM_CONFIG['q2_capacity']}. "
                           f"Expected q1>0, q2>0, and q2<q2_capacity. State: {state}")

        episode_reward = 0.0
        done = False
        episode_events = []

        while not done:
            # State at a decision epoch (env guarantees q1>0 and q2>0 here)
            state_modified, q1_len, q2_len = _map_state_for_training(state)
             # Sanity: decision epoch means both queues > 0 for this server
            if not (q1_len > 0 and q2_len > 0 and q2_len < SIM_CONFIG['q2_capacity']):
                # Advance until a proper decision state appears (or terminal)
                while not (q1_len > 0 and q2_len > 0 and q2_len < SIM_CONFIG['q2_capacity']):
                    env.env.step()
                    env.try_forced_service_on_idle_servers()
                    if env.is_simulation_complete():
                        done = True
                        break
                    if env.current_decision_state is None:
                        continue
                    state = env.get_state_array(env.current_decision_state)
                    state_modified, q1_len, q2_len = _map_state_for_training(state)
                if done:
                    break

            # Let the policy pick; env will enforce forced action if capacity/availability requires it
            action, log_prob, value = agent.select_action(state_modified)

            # Roll env to the *next* decision epoch (or terminal) and accumulate reward
            next_state, cum_reward, done, events = env.step(action, state_modified)

            # If not done, next_state should also be a decision epoch
            if not done and next_state is not None:
                _ns_mod, q1n, q2n = _map_state_for_training(next_state)
                # (Optional) sanity check—comment out if you don't want asserts in prod
                assert q1n > 0 and q2n > 0

            # Log decision & env events (similar to DQN logging)
            for ev in events:
                ev['episode'] = episode
                if ev['event_type'] in ('Decision_ServeQ1', 'Decision_ServeQ2'):
                    ev['current_state'] = state_modified.tolist() if isinstance(state_modified, np.ndarray) else list(state_modified)
                    ev['action'] = int(action)
                    ev['reward'] = float(cum_reward)
                    if not done and next_state is not None:
                        ns_mod, _, _ = _map_state_for_training(next_state)
                        ev['next_state'] = ns_mod.tolist()
                    else:
                        ev['next_state'] = None
                    ev['done'] = bool(done)
                episode_events.append(ev)

            # On-policy storage for PPO
            agent.add_experience(
                state=state_modified,
                action=action,
                reward=cum_reward,
                value=value,
                log_prob=log_prob,
                done=done
            )

            episode_reward += cum_reward
            if not done:
                state = next_state  # next decision epoch

        # Push some raw env events (optional: comment if large)
        env_all_events.extend(env.all_events)
        all_events.extend(episode_events)
        episode_rewards.append(episode_reward)
        agent.episode_count += 1

        # PPO update (every episode or when buffer has enough)
        # NOTE: PPOAgent.update() should compute GAE and clear buffer internally.
        upd = agent.update()
        if isinstance(upd, dict):
            losses.append(upd.get('total_loss', np.nan))
            policy_losses.append(upd.get('policy_loss', np.nan))
            value_losses.append(upd.get('value_loss', np.nan))
            entropy_losses.append(upd.get('entropy_loss', np.nan))

        avg_loss = np.nanmean(losses[-10:]) if losses else 0.0
        print(f"Episode {episode + 1:3d}/{TRAIN_CONFIG['train_num_episodes']} | "
              f"Reward: {episode_reward:8.2f} | "
              f"Events: {len(episode_events):3d} | "
              f"Avg Loss: {avg_loss:.4f}")

        # Save first 3 episodes' detailed CSVs (like your DQN script)
        if episode < 3:
            os.makedirs('output/ppo', exist_ok=True)
            pd.DataFrame(env.all_events).to_csv(f'output/ppo/train_episode_{episode + 1}_all_events.csv', index=False)
            pd.DataFrame(episode_events).to_csv(f'output/ppo/train_episode_{episode + 1}_decision_events.csv', index=False)
            print(f"  Saved episode {episode + 1} all/decision events under output/ppo/")

    return losses, episode_rewards, all_events, env_all_events, policy_losses, value_losses, entropy_losses


# ---------------------------
# Plotting (PPO)
# ---------------------------
def plot_training_curves(losses, rewards, policy_losses=None, value_losses=None, entropy_losses=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Total loss
    axes[0, 0].plot(losses)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Update')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)

    # Episode rewards
    axes[0, 1].plot(rewards)
    axes[0, 1].set_title('Episode Rewards')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True)

    # PPO-specific losses
    if policy_losses:
        axes[1, 0].plot(policy_losses, label='Policy Loss')
        axes[1, 0].plot(value_losses, label='Value Loss')
        axes[1, 0].plot(entropy_losses, label='Entropy Loss')
        axes[1, 0].set_title('PPO Loss Components')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Moving average of rewards
    if len(rewards) > 10:
        window_size = min(50, max(10, len(rewards) // 10))
        moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
        axes[1, 1].plot(rewards, alpha=0.3, label='Raw Rewards')
        axes[1, 1].plot(moving_avg, label=f'Moving Avg (window={window_size})')
        axes[1, 1].set_title('Reward Moving Average')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    os.makedirs('output/ppo', exist_ok=True)
    plt.savefig('output/ppo/ppo_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_events_to_csv(df, filename):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"\nSaved {len(df)} events to {filename}")
        if 'event_type' in df.columns:
            print("\nEvent types distribution:")
            print(df['event_type'].value_counts())
    except Exception as e:
        print(f"Error saving events to CSV: {e}")


# ---------------------------
# Main
# ---------------------------
def main():
    print("Starting PPO Agent Training")
    print("=" * 50)
    print_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Determine state dimension based on config
    if TRAIN_CONFIG.get('modified_states_type') == 'Q1Q2':
        state_dim = 3  # [q1_length, q2_server_length, time]
    elif TRAIN_CONFIG.get('modified_states_type') == 'Q1Q2s':
        state_dim = 3  # [q1_length, q2_current_server, time]
    else:
        # full state (q1 + all q2s + server_id + time) — the network can handle larger dims too,
        # but if you truly use "full", adjust PPOAgent init accordingly.
        # Here we default to 3 to match your usual 'Q1Q2' usage.
        state_dim = 3

    action_dim = NETWORK_CONFIG['action_dim']

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )

    # Optional: load pretrain if your loader is PPO-aware (on-policy buffers differ from DQN)
    # load_and_transfer_pretrained_data(agent)

    losses, rewards, all_events, env_all_events, pol_losses, val_losses, ent_losses = train_ppo_agent(agent)

    # Save model
    os.makedirs('trained_models', exist_ok=True)
    agent.save('trained_models/ppo_trained_model.pth')

    # Plots
    plot_training_curves(losses, rewards, pol_losses, val_losses, ent_losses)

    # Save events
    os.makedirs('output/ppo', exist_ok=True)
    pd.DataFrame(all_events).to_csv('output/ppo/ppo_train_decision_events.csv', index=False)
    pd.DataFrame(env_all_events).to_csv('output/ppo/ppo_train_all_events.csv', index=False)

    print("\nTraining completed!")
    if len(rewards) >= 100:
        print(f"Final avg reward (last 100): {np.mean(rewards[-100:]):.4f}")
    print(f"Total episodes: {len(rewards)}")
    print("Model saved to: trained_models/ppo_trained_model.pth")


if __name__ == "__main__":
    main()
