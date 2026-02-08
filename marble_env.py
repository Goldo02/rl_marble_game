import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from main import MarbleGame

class MarbleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, gui=False, action_repeat=8, max_steps=5000, seed=100):
        super(MarbleEnv, self).__init__()
        
        # Initialize simulation
        self.gui = gui
        self.action_repeat = action_repeat
        self.max_steps = max_steps
        self.current_steps = 0
        self.seed = seed
        
        self.game = MarbleGame(gui=gui, auto_reset=False, seed=seed)
        
        # Action Space: 9 discrete actions
        self.action_space = spaces.Discrete(9)
        
        # Observation Space: 8 continuous values
        low = np.array([-5.0, -5.0, -10.0, -10.0, -0.5, -0.5, -5.0, -5.0], dtype=np.float32)
        high = np.array([5.0, 5.0, 10.0, 10.0, 0.5, 0.5, 5.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Internal state for tilt
        self.tilt_x = 0.0
        self.tilt_y = 0.0
        self.max_tilt = math.radians(10) # 10 degrees
        self.delta_tilt = 0.002 # Per step change (smaller for smoothness)
        
        # Action mapping
        self.action_map = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),  (0, 0),  (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # Target position (locally centered in current simulation)
        self.target_pos = self.game.local_arrival_pos[:2]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.tilt_x = 0.0
        self.tilt_y = 0.0
        self.current_steps = 0
        
        # Step once to settle
        self.game.step(self.tilt_x, self.tilt_y)
        
        # Calculate initial distance
        pos, _ = self.game.get_state()
        self.prev_dist = np.linalg.norm(np.array(pos[:2]) - self.target_pos)
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_steps += 1
        
        # Decode action
        d_tilt_x_idx, d_tilt_y_idx = self.action_map[action]
        
        # Update tilt
        self.tilt_x += d_tilt_x_idx * self.delta_tilt
        self.tilt_y += d_tilt_y_idx * self.delta_tilt
        
        # Clip tilt
        self.tilt_x = np.clip(self.tilt_x, -self.max_tilt, self.max_tilt)
        self.tilt_y = np.clip(self.tilt_y, -self.max_tilt, self.max_tilt)
        
        # Step simulation multiple times (Frame Skipping)
        done = False
        info = {}
        
        for _ in range(self.action_repeat):
            done, info = self.game.step(self.tilt_x, self.tilt_y)
            if done:
                break
        
        obs = self._get_obs()
        
        # Reward Calculation
        dist = info.get('dist', 10.0)
        improvement = self.prev_dist - dist
        self.prev_dist = dist
        
        reward = 100.0 * improvement
        reward -= 0.1 # living penalty
        
        # Terminal rewards
        if done:
            if info.get('cause') == 'win':
                reward += 100.0
            elif info.get('cause') == 'fell':
                reward -= 100.0
        
        terminated = done
        truncated = self.current_steps >= self.max_steps
                
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        pos, vel = self.game.get_state()
        x, y = pos[0], pos[1]
        vx, vy = vel[0], vel[1]
        
        # Calculate distance vector
        dx = x - self.target_pos[0]
        dy = y - self.target_pos[1]
        
        obs = np.array([x, y, vx, vy, self.tilt_x, self.tilt_y, dx, dy], dtype=np.float32)
        return obs

    def close(self):
        pass
