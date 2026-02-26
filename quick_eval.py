import torch
from dqn_agent import DQNAgent
from marble_env import MarbleEnv
import os

def quick_eval():
    checkpoint_dir = "checkpoints"
    best_model_path = os.path.join(checkpoint_dir, "dqn_marble_best.pth")
    
    if not os.path.exists(best_model_path):
        print("No best model found yet.")
        return

    # Use same parameters as train.py
    env = MarbleEnv(gui=False, seed=100)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    print(f"Loading weights from {best_model_path} (State Dim: {state_dim})")
    agent.load(best_model_path)
    
    state, info = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < 1000:
        action_mask = info.get('action_mask', None)
        action = agent.select_action(state, epsilon=0.0, action_mask=action_mask)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
    print(f"\n--- VALUTAZIONE MANUALE ---")
    print(f"Episodio terminato in {steps} passi")
    print(f"Reward Totale: {total_reward:.2f}")
    print(f"Causa fine: {info.get('cause', 'timeout')}")
    
    env.close()

if __name__ == "__main__":
    quick_eval()
