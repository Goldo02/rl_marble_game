import torch
import numpy as np
import pybullet as p
from marble_env import MarbleEnv
from dqn_agent import DQNAgent
import os
import time

def test(checkpoint_path="checkpoints/dqn_marble_best.pth", num_episodes=5):
    # Initialize environment
    # We use a fixed seed for the maze but allow random_spawn if desired
    env = MarbleEnv(gui=True, max_steps=5000, seed=100, random_spawn=False)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = DQNAgent(state_dim, action_dim)
    
    # Load model
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}...")
        agent.load(checkpoint_path)
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Running with random policy.")
    
    # Adjust camera
    p.resetDebugVisualizerCamera(cameraDistance=5.5, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0,0,0])

    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        print(f"\nStarting Episode {ep + 1}")
        
        while not done and steps < 5000:
            # Select action (greedy)
            action = agent.select_action(state, epsilon=0.00)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            if terminated:
                cause = info.get('cause', 'unknown')
                print(f"Episode finished! Cause: {cause.upper()}, Reward: {episode_reward:.1f}, Steps: {steps}")
            elif truncated:
                print(f"Episode truncated (max steps), Reward: {episode_reward:.1f}, Steps: {steps}")

        time.sleep(1) # Gap between episodes

    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dqn_marble_best.pth")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    
    test(checkpoint_path=args.checkpoint, num_episodes=args.episodes)
