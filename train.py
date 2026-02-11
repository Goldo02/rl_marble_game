import torch
import numpy as np
import gymnasium as gym
from marble_env import MarbleEnv
from dqn_agent import DQNAgent
import os
from tqdm.auto import tqdm
from collections import deque
import multiprocessing
from torch.utils.tensorboard import SummaryWriter
from visualize_heatmap import visualize_heatmap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def heatmap_to_image(heatmap_grid, title="Heatmap"):
    """Convert heatmap numpy array to image tensor for TensorBoard"""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(heatmap_grid.T, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar(im, ax=ax, label='Visit Count')
    ax.set_title(title)
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    
    # Convert to numpy array
    fig.canvas.draw()
    # Use buffer_rgba() which is more standard across backends
    rgba_buffer = fig.canvas.buffer_rgba()
    image = np.frombuffer(rgba_buffer, dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # Drop alpha channel
    image = image[:, :, :3]
    plt.close(fig)
    
    # Convert to CHW format for TensorBoard (channels, height, width)
    image = np.transpose(image, (2, 0, 1))
    return image

def run_evaluation(env, agent, num_episodes=10, max_steps=5000):
    """Run deterministic evaluation episodes and return metrics"""
    eval_rewards = []
    eval_successes = []
    eval_steps = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        is_win = False
        
        for _ in range(max_steps):
            # Deterministic action (greedy)
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if terminated or truncated:
                if info.get('cause') == 'win':
                    is_win = True
                break
        
        eval_rewards.append(episode_reward)
        eval_successes.append(1 if is_win else 0)
        eval_steps.append(episode_steps)
        
    avg_reward = sum(eval_rewards) / num_episodes
    success_rate = sum(eval_successes) / num_episodes
    avg_steps = sum(eval_steps) / num_episodes
    
    return avg_reward, success_rate, avg_steps

def train(
    num_episodes=500,
    target_update=2000,  # Now in steps
    save_interval=10,
    log_interval=10,
    epsilon_start=0.6,
    epsilon_decay_episodes=350,
    lr=1e-4,
    gamma=0.99,
    buffer_size=100000,
    batch_size=64,
    success_threshold=0.9, # Stop if 90% success rate
    epsilon_threshold=0.1, # AND epsilon is low enough
    max_steps=2500,
    random_spawn=False,
    checkpoint_dir="checkpoints"
):
    # Print Parameters
    print("\n" + "="*30)
    print("STARTING SINGLE-THREAD TRAINING SESSION")
    print("="*30)
    print(f"Total Target Episodes: {num_episodes}")
    print(f"Max Steps/Episode: {max_steps}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {lr}")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Use the seed currently set in files (100)
    seed = 100
    
    # Create Single Environment
    env = MarbleEnv(gui=False, max_steps=max_steps, seed=seed, random_spawn=random_spawn)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(
        state_dim, 
        action_dim, 
        lr=lr, 
        gamma=gamma, 
        buffer_size=buffer_size, 
        batch_size=batch_size,
        epsilon_start=epsilon_start,
        epsilon_decay_episodes=epsilon_decay_episodes
    )
    
    print(f"Device: {agent.device.type.upper()}")
    print(f"Maze Seed: {seed}")
    
    # Load existing weights if present
    best_model_path = os.path.join(checkpoint_dir, "dqn_marble_best.pth")
    if os.path.exists(best_model_path):
        print(f"Loading existing weights from {best_model_path}...")
        agent.load(best_model_path)
    else:
        print("No existing weights found. Starting from scratch.")
        
    print("="*30 + "\n")
    
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "runs"))
    
    total_steps = 0
    episodes_completed = 0
    best_eval_reward = -float('inf')
    
    # Create a separate environment for evaluation to keep training state clean
    eval_env = MarbleEnv(gui=False, max_steps=max_steps, seed=seed, random_spawn=random_spawn)
    
    # Stats tracking
    rewards_history = deque(maxlen=log_interval)
    steps_history = deque(maxlen=log_interval)
    success_history = deque(maxlen=log_interval)
    
    progress_bar = tqdm(total=num_episodes, desc="Training", position=0, leave=True)
    
    while episodes_completed < num_episodes:
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        is_win = False
        
        episode_buffer = []
        for _ in range(max_steps):
            # 1. NOVELTY-BASED EPSILON:
            # We combine global decay with local novelty.
            # Local epsilon is 1.0 for first visits, decaying as visits increase.
            visits = env.get_current_visits()
            # Formula: 1.0 / (1.0 + ln(visits)) ensures it stays high for a while but eventually decays
            local_novelty_epsilon = 1.0 / (1.0 + np.log(max(1, visits)))
            
            # effective_epsilon is the higher of the two
            effective_epsilon = max(agent.epsilon, local_novelty_epsilon)
            
            # 2. ACTION MASKING: 
            # Get valid actions for current state to prevent hitting walls
            action_mask = env.get_action_mask()
            
            # Select action with local epsilon and mask
            action = agent.select_action(state, epsilon=effective_epsilon, action_mask=action_mask)
            
            # 3. Step env
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 4. Store transition (Initial storage)
            agent.store_transition(state, action, reward, next_state, terminated or truncated)
            # Add to local buffer for oversampling
            episode_buffer.append((state, action, reward, next_state, terminated or truncated))
            
            # Update counters
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            state = next_state
            
            # Log epsilon to TensorBoard occasionally (too noisy for every step)
            if total_steps % 100 == 0:
                writer.add_scalar("Exploration/Effective_Epsilon", effective_epsilon, total_steps)
            
            # 4. Global Update
            loss = agent.update()
            if loss is not None:
                writer.add_scalar("Train/Loss", loss, total_steps)
            
            # Periodic Target Network Update
            if total_steps % target_update == 0:
                agent.update_target_network()
                
            if terminated or truncated:
                if info.get('cause') == 'win':
                    is_win = True
                break
        
        
        # SUCCESS MEMORY: If the episode was a win, store in the special persistent buffer
        if is_win:
            # Add to success buffer (Dual Buffer strategy)
            # We don't need to loop 10 times anymore, just adding it once to the success buffer is sufficient
            # because the sampling strategy ensures it will be used often (50% split).
            for trans in episode_buffer:
                agent.store_success_transition(*trans)
                    
        # Episode finished
        episodes_completed += 1
        progress_bar.update(1)
        
        # Decay epsilon linearly based on global episode count
        agent.decay_epsilon(episodes_completed)
        
        # Log stats
        rewards_history.append(episode_reward)
        steps_history.append(episode_steps)
        success_history.append(1 if is_win else 0)
        
        # Log Global metrics to TensorBoard
        avg_reward = sum(rewards_history) / len(rewards_history) if rewards_history else 0
        avg_steps = sum(steps_history) / len(steps_history) if steps_history else 0
        success_rate = (sum(success_history) / len(success_history)) if success_history else 0
        
        # ADAPTIVE EPSILON: If success rate > 20%, force epsilon <= 0.2
        # Use success_history from the window (10) for recent check
        # We start checking once we have a few episodes (e.g. 5) to avoid noise
        # ADAPTIVE EPSILON: Gradually reduce epsilon as success rate improves
        # This helps reaching the stopping condition faster if the agent is stable.
        if len(success_history) >= 5:
            if success_rate > 0.8:
                agent.epsilon = min(agent.epsilon, 0.05)
            elif success_rate > 0.5:
                agent.epsilon = min(agent.epsilon, 0.1)
            elif success_rate > 0.2:
                agent.epsilon = min(agent.epsilon, 0.2)
        
        writer.add_scalar("Global/Avg_Reward", avg_reward, episodes_completed)
        writer.add_scalar("Global/Success_Rate", success_rate, episodes_completed)
        writer.add_scalar("Global/Avg_Steps", avg_steps, episodes_completed)
        writer.add_scalar("Exploration/Epsilon", agent.epsilon, episodes_completed)
        writer.add_scalar("Episode/Reward", episode_reward, episodes_completed)
        writer.add_scalar("Episode/Steps", episode_steps, episodes_completed)
        
        progress_bar.set_postfix({
            'reward': f"{avg_reward:.1f}",
            'success': f"{success_rate*100:.1f}%",
            'eps': f"{agent.epsilon:.2f}"
        })
        
        if episodes_completed % log_interval == 0:
            # 1. Run Deterministic Evaluation
            eval_reward, eval_success, eval_steps_avg = run_evaluation(eval_env, agent, num_episodes=10, max_steps=max_steps)
            
            writer.add_scalar("Eval/Avg_Reward", eval_reward, episodes_completed)
            writer.add_scalar("Eval/Success_Rate", eval_success, episodes_completed)
            writer.add_scalar("Eval/Avg_Steps", eval_steps_avg, episodes_completed)
            
            tqdm.write(f"\n--- Episode {episodes_completed} EVALUATION ---")
            tqdm.write(f"Eval Reward: {eval_reward:.2f}, Success: {eval_success*100:.1f}%, Epsilon: {agent.epsilon:.3f}")
            
            # 2. Save Best Model based on evaluation performance
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(os.path.join(checkpoint_dir, "dqn_marble_best.pth"))
                tqdm.write(f"New Best Model Saved! (Eval Reward: {eval_reward:.2f})")
            
            # 3. Early Stopping Check based on evaluation success
            if eval_success >= success_threshold and episodes_completed > 50:
                 tqdm.write(f"\n[EARLY STOPPING] Policy solved the maze! (Eval Success: {eval_success*100:.1f}%)")
                 # Break outer loop
                 break
        
        # Periodic Heatmap Export
        if episodes_completed % log_interval == 0:
            try:
                heatmap_grid = env.heatmap_grid
                
                # Save heatmap to disk
                heatmap_path = os.path.join(checkpoint_dir, f'heatmap_ep_{episodes_completed}.png')
                visualize_heatmap(
                    heatmap_grid, 
                    save_path=heatmap_path,
                    title=f'Exploration Heatmap - Episode {episodes_completed}'
                )
                
                # Log heatmap to TensorBoard
                heatmap_image = heatmap_to_image(heatmap_grid, title=f'Heatmap - Ep {episodes_completed}')
                writer.add_image('Heatmap/Global', heatmap_image, episodes_completed)
                
                # --- NEW: Epsilon Heatmap ---
                # Calculate epsilon for every cell based on the same formula used in step()
                # Local novelty part: 1.0 / (1.0 + ln(visits))
                epsilon_grid = 1.0 / (1.0 + np.log(np.maximum(1, heatmap_grid)))
                # Combine with global epsilon
                epsilon_grid = np.maximum(agent.epsilon, epsilon_grid)
                
                # Save epsilon heatmap to disk
                eps_path = os.path.join(checkpoint_dir, f'epsilon_ep_{episodes_completed}.png')
                visualize_heatmap(
                    epsilon_grid, 
                    save_path=eps_path,
                    title=f'Epsilon Heatmap - Episode {episodes_completed}'
                )
                
                # Log epsilon heatmap to TensorBoard
                eps_image = heatmap_to_image(epsilon_grid, title=f'Epsilon Map - Ep {episodes_completed}')
                writer.add_image('Heatmap/Epsilon', eps_image, episodes_completed)
                
            except Exception as e:
                tqdm.write(f"Warning: Could not save heatmap: {e}")


    env.close()
    eval_env.close()
    agent.save(os.path.join(checkpoint_dir, "dqn_marble_final.pth"))
    print("Training finished!")

if __name__ == "__main__":
    train()
