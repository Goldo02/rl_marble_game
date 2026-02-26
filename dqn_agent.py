import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, buffer_size=100000, batch_size=64, 
                 epsilon_start=1.0, epsilon_decay_episodes=1500, persistence=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.success_memory = ReplayBuffer(buffer_size) # Persistent buffer for winning episodes
        
        self.steps_done = 0
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = 0.01
        self.epsilon_decay_episodes = epsilon_decay_episodes
        
        # Sticky Actions (Persistence)
        self.persistence = persistence
        self.sticky_action = None
        self.sticky_steps = 0

    def reset_sticky(self):
        """Reset the persistence state (called at the start of each episode)"""
        self.sticky_action = None
        self.sticky_steps = 0

    def decay_epsilon(self, episode_idx):
        # Linear decay: reaches epsilon_min at epsilon_decay_episodes
        fraction = min(1.0, episode_idx / self.epsilon_decay_episodes)
        self.epsilon = self.epsilon_start - fraction * (self.epsilon_start - self.epsilon_min)
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def select_action(self, state, epsilon=None, action_mask=None):
        self.steps_done += 1
        
        # 1. Check for Sticky Action (Persistence)
        if self.sticky_steps > 0:
            self.sticky_steps -= 1
            # Verify if sticky action is still valid if mask is provided
            if action_mask is not None and not action_mask[self.sticky_action]:
                # Sticky action became invalid! Stop sticking.
                self.sticky_steps = 0
            else:
                return self.sticky_action
        
        eps = epsilon if epsilon is not None else self.epsilon
        
        # Random Action (Exploration)
        if random.random() < eps:
            if action_mask is not None:
                # Choose uniformly from VALID actions
                valid_indices = np.where(action_mask)[0]
                if len(valid_indices) > 0:
                    action = random.choice(valid_indices)
                else:
                    action = random.randrange(self.action_dim)
            else:
                action = random.randrange(self.action_dim)
        
        # Greedy Action (Exploitation)
        else:
            with torch.no_grad():
                # Convert state to tensor safely
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                
                if action_mask is not None:
                    # Set Q-values of invalid actions to negative infinity
                    # Convert mask to tensor 
                    mask_tensor = torch.BoolTensor(action_mask).to(self.device)
                    # We want to set INVALID (False) to -inf. 
                    # So where mask is False ( ~mask_tensor )
                    min_val = float('-inf')
                    q_values[0, ~mask_tensor] = min_val
                
                action = q_values.argmax().item()

        # Activate Persistence for the selected action (Random or Greedy)
        if self.persistence > 1:
            self.sticky_action = action
            self.sticky_steps = self.persistence - 1 # -1 because we execute it now
            
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def store_success_transition(self, state, action, reward, next_state, done):
        """Store transition in the separate success buffer"""
        self.success_memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Dual Buffer Sampling Strategy
        # If we have enough successful experiences, look to sample 50/50
        success_batch_size = 0
        main_batch_size = self.batch_size
        
        if len(self.success_memory) > self.batch_size // 2:
            success_batch_size = self.batch_size // 2
            main_batch_size = self.batch_size - success_batch_size
            
        # Sample from main memory
        state, action, reward, next_state, done = self.memory.sample(main_batch_size)
        
        # Sample from success memory if valid
        if success_batch_size > 0:
            s_state, s_action, s_reward, s_next_state, s_done = self.success_memory.sample(success_batch_size)
            # Concatenate tuples
            state = state + s_state
            action = action + s_action
            reward = reward + s_reward
            next_state = next_state + s_next_state
            done = done + s_done

        # Convert to tensors with high robustness
        def safe_tensor(data, dtype_np, device, is_long=False):
            arr = np.array(data)
            if arr.dtype == object: # Means inconsistent shapes/types in list
                arr = np.stack(data)
            arr = arr.astype(dtype_np)
            t = torch.from_numpy(arr).to(device)
            return t.long() if is_long else t.float()

        state = safe_tensor(state, np.float32, self.device)
        action = safe_tensor(action, np.int64, self.device, is_long=True).unsqueeze(1)
        reward = safe_tensor(reward, np.float32, self.device).unsqueeze(1)
        next_state = safe_tensor(next_state, np.float32, self.device)
        done = safe_tensor(done, np.float32, self.device).unsqueeze(1)

        curr_q = self.policy_net(state).gather(1, action)
        
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + (1 - done) * self.gamma * next_q
            
        loss = nn.MSELoss()(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent explosion with large rewards
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)
        
    def load(self, filepath):
        try:
            self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device, weights_only=True))
        except TypeError: # Older torch version
            self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_checkpoint(self, filepath):
        """Saves everything needed to resume training"""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'memory': list(self.memory.buffer),
            'success_memory': list(self.success_memory.buffer)
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """Loads everything and resumes state"""
        try:
            if not torch.cuda.is_available():
                checkpoint = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
            else:
                checkpoint = torch.load(filepath, weights_only=False)
        except TypeError: # Older torch version
            if not torch.cuda.is_available():
                checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(filepath)
            
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        
        # Restore buffers
        self.memory.buffer.extend(checkpoint['memory'])
        self.success_memory.buffer.extend(checkpoint['success_memory'])
        
        print(f"Checkpoint loaded from {filepath}")
        return self.epsilon
