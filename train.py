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
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def heatmap_to_image(heatmap_grid, title="Heatmap"):
    """Convert heatmap numpy array to image tensor for TensorBoard"""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(heatmap_grid, cmap='hot', interpolation='nearest', origin='upper')
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
        state, info = env.reset()
        agent.reset_sticky()
        episode_reward = 0
        episode_steps = 0
        is_win = False
        
        for _ in range(max_steps):
            # Deterministic action (greedy) with action mask
            action_mask = info.get('action_mask', None)
            action = agent.select_action(state, epsilon=0.0, action_mask=action_mask)
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
    num_episodes=3000,
    target_update=2500,  # Now in steps
    save_interval=100,
    log_interval=100,
    epsilon_start=0.8,
    epsilon_decay_episodes=2200,
    lr=1e-4,
    gamma=0.99,
    buffer_size=100000,
    batch_size=64,
    success_threshold=0.9, 
    epsilon_threshold=0.1,
    max_steps=5000,
    random_spawn=False,
    seed=100,
    gui=True,
    checkpoint_dir="checkpoints",
    persistence=5,
    resume=None
):
    # Print Parameters
    print("\n" + "="*40)
    print("STARTING SINGLE-THREAD TRAINING SESSION")
    print("="*40)
    print(f"  Episodes:          {num_episodes}")
    print(f"  Max Steps/Episode: {max_steps}")
    print(f"  Batch Size:        {batch_size}")
    print(f"  Learning Rate:     {lr}")
    print(f"  Gamma:             {gamma}")
    print(f"  Buffer Size:       {buffer_size}")
    print(f"  Target Update:     {target_update} steps")
    print(f"  Epsilon Start:     {epsilon_start}")
    print(f"  Epsilon Decay:     {epsilon_decay_episodes} episodes")
    print(f"  Success Threshold: {success_threshold}")
    print(f"  Persistence:       {persistence}")
    print(f"  Random Spawn:      {random_spawn}")
    print(f"  Seed:              {seed}")
    print(f"  GUI:               {gui}")
    print(f"  Checkpoint Dir:    {checkpoint_dir}")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Create Single Environment
    env = MarbleEnv(gui=gui, max_steps=max_steps, seed=seed, random_spawn=random_spawn)
    
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
        epsilon_decay_episodes=epsilon_decay_episodes,
        persistence=persistence
    )
    
    print(f"Device: {agent.device.type.upper()}")
    print(f"Maze Seed: {seed}")
    
    # Load from checkpoint if resume is specified
    if resume:
        checkpoint_path = resume if os.path.isfile(resume) else os.path.join(checkpoint_dir, resume)
        if os.path.exists(checkpoint_path):
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            agent.load_checkpoint(checkpoint_path)
            
            # Extract episode number from filename if possible (e.g., dqn_checkpoint_ep_50.pth)
            import re
            ep_match = re.search(r'ep_(\d+)', os.path.basename(checkpoint_path))
            if ep_match:
                episodes_completed = int(ep_match.group(1))
                print(f"Resuming from episode {episodes_completed}")
            
            # Load heatmap if it exists
            heatmap_path = checkpoint_path.replace('.pth', '_heatmap.npy')
            if os.path.exists(heatmap_path):
                env.load_heatmap(heatmap_path)
                print(f"Loaded heatmap from {heatmap_path}")
        else:
            print(f"Error: Checkpoint {checkpoint_path} not found!")
    else:
        # Fallback to loading best weights if not resuming
        best_model_path = os.path.join(checkpoint_dir, "dqn_marble_best.pth")
        if os.path.exists(best_model_path):
            print(f"Loading existing best weights from {best_model_path}...")
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
    
    # NEW: For Delta Heatmap tracking
    previous_heatmap_grid = np.zeros_like(env.heatmap_grid)
    
    progress_bar = tqdm(total=num_episodes, desc="Training", position=0, leave=True)
    
    while episodes_completed < num_episodes:
        state, info = env.reset() # Capture info from reset
        agent.reset_sticky()
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
            local_novelty_epsilon = 0.8 / (1.0 + np.log(max(1, visits)))
            
            # DAMPENING: As the global agent epsilon decays, we also reduce the influence 
            # of local novelty to ensure the policy eventually stabilizes (convergence).
            novelty_factor = agent.epsilon / agent.epsilon_start
            effective_epsilon = max(agent.epsilon, local_novelty_epsilon * novelty_factor)
        
            
            # Retrieve Action Mask from info (available from both reset and step)
            action_mask = info.get('action_mask', None)
            
            # Select action with local epsilon
            action = agent.select_action(state, epsilon=effective_epsilon, action_mask=action_mask)
            # print("Action: ", action)
            
            # 3. Step env
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 4. Store transition (Initial storage)
            agent.store_transition(state, action, reward, next_state, terminated or truncated)
            # Add to local buffer for oversampling
            episode_buffer.append((state, action, reward, next_state, terminated or truncated))
            
            
            # Update counters FIRST
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            
            # Update HUD Metrics if GUI is enabled (throttled to every 10 steps for performance)
            if env.gui and episode_steps > 0 and episode_steps % 10 == 0:
                env.set_debug_metrics(
                    episode=episodes_completed + 1, 
                    epsilon=effective_epsilon, 
                    reward=episode_reward,
                    stagnation=info.get('stagnation_counter', 0),
                    velocity=info.get('velocity', 0.0),
                    steps=episode_steps
                )
            
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
                agent.epsilon = min(agent.epsilon, 0.01)
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
            
            # 2. Save Best Model weights
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(os.path.join(checkpoint_dir, "dqn_marble_best.pth"))
                tqdm.write(f"New Best Model Saved! (Eval Reward: {eval_reward:.2f})")
            
            # 3. Save Full Checkpoint for resumption
            checkpoint_path = os.path.join(checkpoint_dir, f"dqn_checkpoint_ep_{episodes_completed}.pth")
            agent.save_checkpoint(checkpoint_path)
            env.save_heatmap(checkpoint_path.replace('.pth', '_heatmap.npy'))
            
            # 4. Early Stopping Check based on evaluation success
            if eval_success >= success_threshold and episodes_completed > 50:
                 tqdm.write(f"\n[EARLY STOPPING] Policy solved the maze! (Eval Success: {eval_success*100:.1f}%)")
                 # Break outer loop
                 break
        
        if episodes_completed % log_interval == 0:
            try:
                current_heatmap = env.heatmap_grid.copy()
                
                # 1. Standard Heatmap
                heatmap_path = os.path.join(checkpoint_dir, f'heatmap_ep_{episodes_completed}.png')
                visualize_heatmap(current_heatmap, save_path=heatmap_path, title=f'Heatmap - Episode {episodes_completed}')
                
                # 2. Logarithmic Heatmap (Makes low-visit areas visible)
                log_heatmap = np.log1p(current_heatmap)
                log_path = os.path.join(checkpoint_dir, f'heatmap_log_ep_{episodes_completed}.png')
                visualize_heatmap(log_heatmap, save_path=log_path, title=f'Log Heatmap (Detailed) - Ep {episodes_completed}')
                
                # 3. Delta Heatmap (Recent activity only)
                delta_heatmap = current_heatmap - previous_heatmap_grid
                delta_path = os.path.join(checkpoint_dir, f'heatmap_delta_ep_{episodes_completed}.png')
                visualize_heatmap(delta_heatmap, save_path=delta_path, title=f'Recent Activity (Delta) - Ep {episodes_completed}')
                
                # Update previous for next interval
                previous_heatmap_grid = current_heatmap
                
                # 4. Epsilon Map
                epsilon_grid = 1.0 / (1.0 + np.log(np.maximum(1, current_heatmap)))
                epsilon_grid = np.maximum(agent.epsilon, epsilon_grid)
                eps_path = os.path.join(checkpoint_dir, f'epsilon_ep_{episodes_completed}.png')
                visualize_heatmap(epsilon_grid, save_path=eps_path, title=f'Epsilon Map - Ep {episodes_completed}')
                
                # Log to TensorBoard
                writer.add_image('Heatmap/Standard', heatmap_to_image(current_heatmap), episodes_completed)
                writer.add_image('Heatmap/Logarithmic', heatmap_to_image(log_heatmap), episodes_completed)
                writer.add_image('Heatmap/Delta', heatmap_to_image(delta_heatmap), episodes_completed)
                writer.add_image('Heatmap/Epsilon', heatmap_to_image(epsilon_grid), episodes_completed)
                
            except Exception as e:
                tqdm.write(f"Warning: Could not save heatmap: {e}")


    env.close()
    eval_env.close()
    agent.save(os.path.join(checkpoint_dir, "dqn_marble_final.pth"))
    print("Training finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Agent for Marble Game")
    parser.add_argument("--num_episodes", type=int, default=2000, help="Total episodes to train")
    parser.add_argument("--max_steps", type=int, default=5000, help="Max steps per episode")
    parser.add_argument("--no-gui", action="store_false", dest="gui", default=True, help="Disable PyBullet GUI (enabled by default)")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--random_spawn", action="store_true", help="Randomize marble spawn point")
    parser.add_argument("--persistence", type=int, default=5, help="Number of steps to repeat random actions")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    
    train(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        gui=args.gui,
        seed=args.seed,
        lr=args.lr,
        batch_size=args.batch_size,
        random_spawn=args.random_spawn,
        persistence=args.persistence,
        resume=args.resume
    )
