import torch
from dqn_agent import DQNAgent
from marble_env import MarbleEnv
import os
import numpy as np

def quick_eval(checkpoint_name):
    checkpoint_dir = "checkpoints"
    model_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found.")
        return

    # Use same parameters as train.py
    # Seed 100 for consistency with previous tests
    env = MarbleEnv(gui=False, seed=100)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    print(f"Loading from {model_path}")
    
    if "checkpoint" in checkpoint_name:
        agent.load_checkpoint(model_path)
    else:
        agent.load(model_path)
    
    state, info = env.reset()
    agent.reset_sticky() # Ensure persistence starts clean
    
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < 2500:
        action_mask = info.get('action_mask', None)
        # Epsilon 0.0 tests the new Greedy Persistence
        action = agent.select_action(state, epsilon=0.0, action_mask=action_mask)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
    print(f"\n--- VALUTAZIONE POST-FIX ---")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Episodio terminato in {steps} passi")
    print(f"Reward Totale: {total_reward:.2f}")
    print(f"Causa fine: {info.get('cause', 'timeout')}")
    
    env.close()

if __name__ == "__main__":
    # Test a checkpoint that was previously "stalled" at -520
    quick_eval("dqn_checkpoint_ep_1400.pth")
