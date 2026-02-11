import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from marble_game import MarbleGame

class ExplorationHeatmap:
    def __init__(self, size=20, bounds=2.0):
        self.size = size
        self.bounds = bounds
        self.grid = np.zeros((size, size), dtype=int)
        self.episode_visited = set()  # Track visits in current episode
        
    def get_indices(self, x, y):
        # Map [-bounds, bounds] to [0, size-1]
        i = int((x + self.bounds) / (2 * self.bounds) * self.size)
        j = int((y + self.bounds) / (2 * self.bounds) * self.size)
        return np.clip(i, 0, self.size - 1), np.clip(j, 0, self.size - 1)
    
    def update(self, x, y):
        i, j = self.get_indices(x, y)
        is_new = False
        # Use [j, i] (y, x) indexing to match maze grid structure
        if (j, i) not in self.episode_visited:
            self.grid[j, i] += 1
            self.episode_visited.add((j, i))
            is_new = True
        return self.grid[j, i], is_new

    def reset_episode(self):
        """Reset episode-specific visitation tracker"""
        self.episode_visited.clear()

    def decay_visitation(self, x, y, factor=0.5):
        """Reduces visitation count in a specific area (e.g., after a failure)"""
        i, j = self.get_indices(x, y)
        self.grid[i, j] = int(self.grid[i, j] * factor)

    def global_decay(self, factor=0.99):
        """Slowly reduces all visitation counts to keep exploration dynamic"""
        self.grid = (self.grid * factor).astype(int)

class MarbleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, gui=False, action_repeat=8, max_steps=2500, seed=100, random_spawn=False):
        super(MarbleEnv, self).__init__()
        
        # Initialize simulation
        self.gui = gui
        self.action_repeat = action_repeat
        self.max_steps = max_steps
        self.current_steps = 0
        self.seed = seed
        self.random_spawn = random_spawn
        
        self.game = MarbleGame(gui=gui, auto_reset=False, seed=seed)
        self.heatmap = ExplorationHeatmap(size=20, bounds=2.0)
        
        # Pre-calculate BFS distance map for guidance
        self.bfs_map = self.game.maze_gen.get_bfs_distance_map()
        
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
        self.delta_tilt = 0.01 # Per step change (more responsive)
        
        # Action mapping
        self.action_map = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),  (0, 0),  (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # Target position (locally centered in current simulation)
        self.target_pos = self.game.local_arrival_pos[:2]

    def get_grid_indices(self, x, y):
        """Helper to map world coordinates to maze grid indices"""
        offset_x = -(self.game.maze_width * self.game.cell_size) / 2
        offset_y = -(self.game.maze_height * self.game.cell_size) / 2
        
        gx = int((x - offset_x) / self.game.cell_size)
        gy = int((y - offset_y) / self.game.cell_size)
        
        return np.clip(gx, 0, self.game.maze_width - 1), np.clip(gy, 0, self.game.maze_height - 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(random_spawn=self.random_spawn)
        self.tilt_x = 0.0
        self.tilt_y = 0.0
        self.current_steps = 0
        self.stagnation_counter = 0  # Track consecutive low-velocity steps
        
        # Apply a small global decay to heatmap at each reset to simulate 'fading memory'
        # and prevent exploration from permanently stalling in any area
        self.heatmap.global_decay(0.99)
        
        # Step once to settle
        self.game.step(self.tilt_x, self.tilt_y)
        
        # Calculate initial distance
        pos, _ = self.game.get_state()
        self.prev_dist = np.linalg.norm(np.array(pos[:2]) - self.target_pos)
        
        # Initialize BFS distance tracking
        gx, gy = self.get_grid_indices(pos[0], pos[1])
        # Use .get() or handle array safely, but we know it's a numpy array [y, x]
        self.prev_bfs_dist = self.bfs_map[gy][gx]
        
        # In case we start in a 999 cell (wall/hole), cap it to a reasonable high value
        if self.prev_bfs_dist >= 999:
            self.prev_bfs_dist = 100 
            
        if hasattr(self, 'prev_dist_to_local'):
            delattr(self, 'prev_dist_to_local')
            
        # Reset episode-specific visitation tracker
        self.heatmap.reset_episode()
        
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
        # Reward Calculation
        dist = info.get('dist', 10.0)
        
        # --- NEW CONTINUOUS BFS REWARD LOGIC ---
        ball_pos, ball_vel = self.game.get_state()
        velocity_magnitude = np.linalg.norm(ball_vel[:2])
        gx, gy = self.get_grid_indices(ball_pos[0], ball_pos[1])
        
        # 1. Find the best neighbor (lowest BFS distance)
        best_neighbor_dist = float('inf')
        best_nx, best_ny = gx, gy
        
        current_bfs_dist = self.bfs_map[gy][gx]
        
        # In case the ball fell or is in a hole where BFS is 999 (Wall) or 2 (Hole)
        # Ensure current_bfs_dist is a valid number for comparison
        if not np.isfinite(current_bfs_dist):
            current_bfs_dist = 999 # Treat as a very high distance if invalid
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = gx + dx, gy + dy
            if 0 <= nx < self.game.maze_width and 0 <= ny < self.game.maze_height:
                d = self.bfs_map[ny][nx]
                # Ensure neighbor distance is also finite for comparison
                if not np.isfinite(d):
                    d = 999
                if d < best_neighbor_dist:
                    best_neighbor_dist = d
                    best_nx, best_ny = nx, ny
        
        # 2. If we are in the target cell (BFS=0), target is the actual goal position
        # Otherwise, target is the center of the next best cell
        if current_bfs_dist == 0:
            target_x, target_y = self.target_pos[0], self.target_pos[1]
        else:
            # Calculate world coordinates of the center of best_nx, best_ny
            offset_x = -(self.game.maze_width * self.game.cell_size) / 2
            offset_y = -(self.game.maze_height * self.game.cell_size) / 2
            target_x = offset_x + (best_nx * self.game.cell_size) + (self.game.cell_size / 2)
            target_y = offset_y + (best_ny * self.game.cell_size) + (self.game.cell_size / 2)
            
        # 3. Calculate distance to this immediate LOCAL target
        dist_to_local_target = np.linalg.norm([ball_pos[0] - target_x, ball_pos[1] - target_y])
        
        # 4. Continuous Reward:
        # - Discrete improvement (jumping cells): (prev_bfs - curr_bfs) * 1.0
        # - Continuous improvement (getting closer to center of next cell): (prev_local_dist - curr_local_dist) * 10.0
        
        # BFS Reward Logic: Standardize for coordinate system and prevent farming
        # MazeGenerator uses dist_map[y][x], so we use self.bfs_map[gy][gx]
        gx, gy = self.get_grid_indices(ball_pos[0], ball_pos[1])
        current_bfs_raw = self.bfs_map[gy][gx]
        
        # 1. Coordinate Robustness: If outside or in a wall/hole, current_bfs is very high
        if current_bfs_raw >= 999:
            # We don't want to update prev_bfs_dist with 999, as that triggers a jump reward later
            # Instead, we just give a small BFS penalty for being off-path
            potential_reward = -0.1
        else:
            # 2. Path Movement: Calculate discrete reward for cell changes
            # We only give reward if current_bfs is a valid path distance
            potential_reward = (self.prev_bfs_dist - current_bfs_raw) * 5.0
            
            # 3. Prevent Farming: Only update prev_bfs_dist if we are on a valid path
            self.prev_bfs_dist = current_bfs_raw

        # Add safety clip for any unexpected jumps
        potential_reward = np.clip(potential_reward, -25.0, 25.0)

        # Continuous Micro-Reward (Improvement towards local target)
        # We initialize prev_local_dist if it doesn't exist (first step)
        if not hasattr(self, 'prev_dist_to_local'):
            self.prev_dist_to_local = dist_to_local_target
            
        local_improvement = self.prev_dist_to_local - dist_to_local_target
        self.prev_dist_to_local = dist_to_local_target
        
        # Weighted sum: Big reward for cell change, small reward for movement within cell
        extrinsic_reward = potential_reward + (local_improvement * 1.0)
        
        # Removed living penalty to avoid accumulating large negative rewards
        
        # Stagnation Penalty: Detect if ball is stuck (low velocity)
        if velocity_magnitude < 0.05:  # Threshold for "stuck"
            self.stagnation_counter += 1
            # Constant penalty to avoid quadratic explosion
            # Adjusted for longer-term survival: -0.01
            stagnation_penalty = -0.01
            extrinsic_reward += stagnation_penalty
        else:
            self.stagnation_counter = 0  # Reset if moving
        
        # Intrinsic Discovery Reward (Increased base from 0.05 to 2.0)
        # FIX: Only grant if it's the FIRST time we visit this cell in this episode
        visits, is_new_visit = self.heatmap.update(ball_pos[0], ball_pos[1])
        intrinsic_reward = (2.0 / visits) if is_new_visit else 0.0
        
        reward = extrinsic_reward + intrinsic_reward
        
        info['visits'] = visits
        info['intrinsic_reward'] = intrinsic_reward
        info['extrinsic_reward'] = extrinsic_reward
        info['stagnation_counter'] = self.stagnation_counter
        
        # Terminal rewards
        if done:
            if info.get('cause') == 'win':
                # Massive win bonus
                # Increased to 50.0 to make it the primary target
                win_bonus = 50.0
                reward += win_bonus
                extrinsic_reward += win_bonus
                
                # Speed bonus: reward faster solutions
                # Old: 5000.0 -> New: 1.0
                speed_bonus = 1.0 * (1.0 - self.current_steps / self.max_steps)
                reward += speed_bonus
                extrinsic_reward += speed_bonus
                
            elif info.get('cause') == 'fell':
                # Increased Fall Penalty (was -5.0)
                fall_penalty = -20.0
                
                reward += fall_penalty
                extrinsic_reward += fall_penalty
                
                # RECOVERY LOGIC: If we fell, we want to RESET the visitation count here
                # so that epsilon goes back UP and the agent tries something different next time
                self.heatmap.decay_visitation(ball_pos[0], ball_pos[1], factor=0.1) # Aggressive reduction
        
        terminated = done
        truncated = self.current_steps >= self.max_steps
        
        # Truncation penalty: discourage reaching max steps without winning
        if truncated and not terminated:
            truncation_penalty = 0.0 # Neutralized to prevent suicide behavior
            reward += truncation_penalty
            extrinsic_reward += truncation_penalty
        
        # CLAMP REWARD: Force reward to be in [-50, 50] range
        reward = np.clip(reward, -50.0, 50.0)
                
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        pos, vel = self.game.get_state()
        x, y = pos[0], pos[1]
        vx, vy = vel[0], vel[1]
        
        # Calculate distance vector
        dx = x - self.target_pos[0]
        dy = y - self.target_pos[1]
        
        # Normalize observations to approximately [-1, 1]
        # X, Y: [-5, 5] -> [-1, 1]
        # VX, VY: [-10, 10] -> [-1, 1]
        # Tilt: [-0.17, 0.17] (10 deg) -> [-0.34, 0.34] if divided by 0.5
        # Distance: [-5, 5] -> [-1, 1]
        
        obs = np.array([
            x / 5.0, 
            y / 5.0, 
            vx / 10.0, 
            vy / 10.0, 
            self.tilt_x / 0.5, 
            self.tilt_y / 0.5, 
            dx / 5.0, 
            dy / 5.0
        ], dtype=np.float32)
        return obs

    def close(self):
        # MarbleGame handles its own cleanup via PyBullet disconnect
        pass

    @property
    def heatmap_grid(self):
        return self.heatmap.grid

    def get_current_visits(self):
        """Returns the visitation count of the current ball position"""
        pos, _ = self.game.get_state()
        i, j = self.heatmap.get_indices(pos[0], pos[1])
        return self.heatmap.grid[j, i]

    def get_action_mask(self):
        """
        Returns a boolean mask of valid actions.
        True means the action is valid, False means it would result in a wall collision.
        """
        pos, _ = self.game.get_state()
        gx, gy = self.get_grid_indices(pos[0], pos[1])
        mask = np.ones(self.action_space.n, dtype=bool)
        
        for i, (dx_idx, dy_idx) in enumerate(self.action_map):
            # Calculate target grid cell for this action
            # Note: dx_idx is change in TILT, which roughly corresponds to acceleration direction
            # For simplicity in masking, we look at immediate neighbors in that direction
            nx, ny = gx + dx_idx, gy + dy_idx
            
            # Check bounds and walls
            is_valid = False
            if 0 <= nx < self.game.maze_width and 0 <= ny < self.game.maze_height:
                cell_value = self.game.maze_gen.grid[ny][nx]
                if cell_value != 1: # Not a Wall
                    is_valid = True
            
            mask[i] = is_valid
                    
        return mask
