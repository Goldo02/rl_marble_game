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
                 epsilon_start=1.0, epsilon_decay_episodes=1500):
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
        self.epsilon_min = 0.05
        self.epsilon_decay_episodes = epsilon_decay_episodes

    def decay_epsilon(self, episode_idx):
        # Linear decay: reaches epsilon_min at epsilon_decay_episodes
        fraction = min(1.0, episode_idx / self.epsilon_decay_episodes)
        self.epsilon = self.epsilon_start - fraction * (self.epsilon_start - self.epsilon_min)
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def select_action(self, state, epsilon=None, action_mask=None):
        self.steps_done += 1
        
        eps = epsilon if epsilon is not None else self.epsilon
        
        # Random Action (Exploration)
        if random.random() < eps:
            if action_mask is not None:
                # Choose uniformly from VALID actions
                valid_indices = np.where(action_mask)[0]
                if len(valid_indices) > 0:
                    return random.choice(valid_indices)
                return random.randrange(self.action_dim) # Fallback if all masked (shouldn't happen)
            else:
                return random.randrange(self.action_dim)
        
        # Greedy Action (Exploitation)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                
                if action_mask is not None:
                    # Set Q-values of invalid actions to negative infinity
                    # Convert mask to tensor 
                    mask_tensor = torch.BoolTensor(action_mask).to(self.device)
                    # We want to set INVALID (False) to -inf. 
                    # So where mask is False ( ~mask_tensor )
                    min_val = float('-inf')
                    q_values[0, ~mask_tensor] = min_val
                
                return q_values.argmax().item()

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

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

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
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
