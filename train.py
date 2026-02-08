import torch
import numpy as np
import gymnasium as gym
from marble_env import MarbleEnv
from dqn_agent import DQNAgent
import os
from tqdm.auto import tqdm
from collections import deque
import multiprocessing

def make_env(seed, gui=False, max_steps=5000, random_spawn=False):
    def _init():
        env = MarbleEnv(gui=gui, max_steps=max_steps, seed=seed, random_spawn=random_spawn)
        return env
    return _init

def train(
    num_episodes=2000,
    target_update=10,
    save_interval=50,
    log_interval=10,
    lr=1e-4,
    gamma=0.99,
    buffer_size=10000,
    batch_size=128,
    success_threshold=0.9, # Stop if 90% success rate
    max_steps=2500,
    random_spawn=True,
    checkpoint_dir="checkpoints"
):
    # Detect CPU threads and use n-2 (min 1)
    cpu_count = os.cpu_count() or 1
    num_envs = max(1, cpu_count - 2)
    
    # Print Parameters
    print("\n" + "="*30)
    print("STARTING PARALLEL TRAINING SESSION")
    print("="*30)
    print(f"Num Parallel Envs: {num_envs}")
    print(f"Total Target Episodes: {num_episodes}")
    print(f"Max Steps/Episode: {max_steps}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {lr}")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Use the seed currently set in files (100)
    seed = 100
    
    # Create Vectorized Environment
    env_fns = [make_env(seed, gui=False, max_steps=max_steps, random_spawn=random_spawn) for _ in range(num_envs)]
    envs = gym.vector.AsyncVectorEnv(env_fns)
    
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n
    
    agent = DQNAgent(
        state_dim, 
        action_dim, 
        lr=lr, 
        gamma=gamma, 
        buffer_size=buffer_size, 
        batch_size=batch_size
    )
    
    print(f"Device: {agent.device.type.upper()}")
    print(f"Maze Seed: {seed}")
    print("="*30 + "\n")
    
    total_steps = 0
    episodes_completed = 0
    best_avg_reward = -float('inf')
    
    # Stats tracking
    rewards_history = deque(maxlen=log_interval)
    steps_history = deque(maxlen=log_interval)
    success_history = deque(maxlen=log_interval)
    
    # Track current episode stats for each environment
    env_rewards = np.zeros(num_envs)
    env_steps = np.zeros(num_envs)
    
    states, _ = envs.reset()
    
    progress_bar = tqdm(total=num_episodes, desc="Training", position=0, leave=True)
    
    while episodes_completed < num_episodes:
        # Calculate diversified epsilons for each environment
        # We vary epsilon between agent.epsilon and agent.epsilon^x to have some envs 
        # exploring more and others exploiting more.
        env_epsilons = [agent.epsilon**(1 + i/(num_envs-1) * 2) if num_envs > 1 else agent.epsilon for i in range(num_envs)]
        
        # 1. Select actions for all envs
        actions = []
        for i in range(num_envs):
            actions.append(agent.select_action(states[i], epsilon=env_epsilons[i]))
        actions = np.array(actions)
        
        # 2. Step all envs
        next_states, rewards, terminated, truncated, infos = envs.step(actions)
        
        # 3. Store transitions and update counters
        for i in range(num_envs):
            agent.store_transition(states[i], actions[i], rewards[i], next_states[i], terminated[i] or truncated[i])
            env_rewards[i] += rewards[i]
            env_steps[i] += 1
            
            if terminated[i] or truncated[i]:
                # Episode finished in env i
                episodes_completed += 1
                progress_bar.update(1)
                
                # Decay epsilon only when an episode finishes
                agent.decay_epsilon()
                
                # Log stats
                rewards_history.append(env_rewards[i])
                steps_history.append(env_steps[i])
                
                # Check win condition from info
                is_win = False
                if 'final_info' in infos and infos['final_info'][i] is not None:
                    if infos['final_info'][i].get('cause') == 'win':
                        is_win = True
                
                success_history.append(1 if is_win else 0)
                
                # Reset local counters for this env
                env_rewards[i] = 0
                env_steps[i] = 0
                
                if episodes_completed % log_interval == 0:
                    avg_reward = sum(rewards_history) / len(rewards_history) if rewards_history else 0
                    avg_steps = sum(steps_history) / len(steps_history) if steps_history else 0
                    success_rate = (sum(success_history) / len(success_history)) if success_history else 0
                    
                    progress_bar.set_postfix({
                        'reward': f"{avg_reward:.1f}",
                        'success': f"{success_rate*100:.1f}%",
                        'eps': f"{agent.epsilon:.2f}"
                    })
                    
                    if (episodes_completed) % (log_interval * 5) == 0:
                        tqdm.write(f"\n--- Episodes {episodes_completed} Stats ---")
                        tqdm.write(f"Avg Reward: {avg_reward:.2f}, Success: {success_rate*100:.1f}%, Epsilon: {agent.epsilon:.3f}")
                        
                        if avg_reward > best_avg_reward:
                            best_avg_reward = avg_reward
                            agent.save(os.path.join(checkpoint_dir, "dqn_marble_best.pth"))

        # 4. Global Update
        agent.update()
        total_steps += 1
        
        if total_steps % (target_update * 10) == 0: # Adjusted frequency for parallel envs
            agent.update_target_network()
            
        states = next_states

        # Early Stopping check
        if len(success_history) >= log_interval:
            success_rate = sum(success_history) / len(success_history)
            if success_rate >= success_threshold and episodes_completed > 50:
                 tqdm.write(f"\n[EARLY STOPPING] Success rate {success_rate*100:.1f}% reached!")
                 break

    envs.close()
    agent.save(os.path.join(checkpoint_dir, "dqn_marble_final.pth"))
    print("Training finished!")

if __name__ == "__main__":
    train()
