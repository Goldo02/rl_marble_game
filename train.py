import torch
import numpy as np
from marble_env import MarbleEnv
from dqn_agent import DQNAgent
import os
from tqdm.auto import tqdm
from collections import deque

def train(
    num_episodes=500,
    target_update=10,
    save_interval=50,
    log_interval=10,
    lr=1e-4,
    gamma=0.99,
    buffer_size=10000,
    batch_size=64,
    success_threshold=0.9, # Stop if 90% success rate
    max_steps=5000,
    checkpoint_dir="checkpoints"
):
    # Print Parameters
    print("\n" + "="*30)
    print("STARTING TRAINING SESSION")
    print("="*30)
    print(f"Episodes: {num_episodes}")
    print(f"Max Steps/Episode: {max_steps}")
    print(f"Target Update: {target_update}")
    print(f"Save Interval: {save_interval}")
    print(f"Log Interval: {log_interval}")
    print(f"Success Threshold: {success_threshold*100}%")
    print(f"Learning Rate: {lr}")
    print(f"Gamma: {gamma}")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    env = MarbleEnv(gui=False, max_steps=max_steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(
        state_dim, 
        action_dim, 
        lr=lr, 
        gamma=gamma, 
        buffer_size=buffer_size, 
        batch_size=batch_size
    )
    
    print(f"Device: {agent.device.type.upper()}")
    print("="*30 + "\n")
    
    total_steps = 0
    best_avg_reward = -float('inf')
    
    # Stats tracking for logging
    rewards_history = deque(maxlen=log_interval)
    steps_history = deque(maxlen=log_interval)
    success_history = deque(maxlen=log_interval)
    
    progress_bar = tqdm(range(num_episodes), desc="Training", position=0, leave=True)
    
    for episode in progress_bar:
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Nested progress bar for episode steps with fixed position
        # episode_bar = tqdm(total=max_steps, desc=f"  Step", leave=False, unit="step", position=1)
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            # episode_bar.update(1)
            
        # episode_bar.close()
        agent.decay_epsilon()
        
        # Track stats
        rewards_history.append(episode_reward)
        steps_history.append(episode_steps)
        success_history.append(1 if info.get('cause') == 'win' else 0)
        
        if episode % target_update == 0:
            agent.update_target_network()
            
        # Update progress bar info
        progress_bar.set_postfix({
            'reward': f"{episode_reward:.1f}",
            'eps': f"{agent.epsilon:.2f}"
        })
            
        if (episode + 1) % log_interval == 0:
            avg_reward = sum(rewards_history) / len(rewards_history)
            avg_steps = sum(steps_history) / len(steps_history)
            success_rate = (sum(success_history) / len(success_history))
            
            tqdm.write(f"\n--- Episode {episode+1} Stats ---")
            tqdm.write(f"Avg Reward (last {log_interval}): {avg_reward:.2f}")
            tqdm.write(f"Avg Steps (last {log_interval}): {avg_steps:.1f}")
            tqdm.write(f"Success Rate (last {log_interval}): {success_rate*100:.1f}%")
            tqdm.write(f"Current Epsilon: {agent.epsilon:.3f}")
            
            # Save Best Model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(os.path.join(checkpoint_dir, "dqn_marble_best.pth"))
                tqdm.write(f"*** New Best Model Saved (Avg Reward: {best_avg_reward:.2f}) ***")

            # Early Stopping
            if success_rate >= success_threshold and episode > 50:
                tqdm.write(f"\n[EARLY STOPPING] Success rate {success_rate*100:.1f}% reached threshold!")
                tqdm.write("The agent has learned to beat the game effectively.")
                break
                
            tqdm.write("-" * 25)
            
        if episode % save_interval == 0 and episode > 0:
            agent.save(os.path.join(checkpoint_dir, f"dqn_marble_ep{episode}.pth"))

    agent.save(os.path.join(checkpoint_dir, "dqn_marble_final.pth"))
    print("Training finished!")

if __name__ == "__main__":
    train()
