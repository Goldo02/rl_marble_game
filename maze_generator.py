import random
from collections import deque
import numpy as np

class MazeGenerator:
    """
    Generates a maze using DFS and places holes in 'spiazzi' (open areas).
    0 = Path
    1 = Wall
    2 = Hole
    3 = Arrival (Goal)
    4 = Start
    """
    def __init__(self, width, height, seed=None):
        # Ensure dimensions are odd for DFS wall carving
        self.width = width if width % 2 != 0 else width + 1
        self.height = height if height % 2 != 0 else height + 1
        self.grid = [[1 for _ in range(self.width)] for _ in range(self.height)]
        self.rng = random.Random(seed)
        
    def generate(self):
        # 1. Carve maze using DFS
        self._carve_dfs(1, 1)
        
        # 2. Add Arrival zone (Bottom-Right, but ensuring it's on a path)
        # Find a suitable spot for arrival near the end
        self.grid[self.height - 2][self.width - 2] = 3
        
        # 3. Add Explicit Start point at (1, 1) - Top Left
        self.grid[1][1] = 4
        
        # 4. Add Holes in "Spiazzi"
        # We'll add a few holes, each surrounded by a 3x3 open area
        num_holes = max(5, (self.width * self.height) // 50)
        self._add_holes_in_spiazzi(num_holes)
        
        return self.grid

    def _carve_dfs(self, x, y):
        self.grid[y][x] = 0
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        self.rng.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < self.width - 1 and 0 < ny < self.height - 1:
                if self.grid[ny][nx] == 1:
                    # Remove wall between current and neighbor
                    self.grid[y + dy // 2][x + dx // 2] = 0
                    self._carve_dfs(nx, ny)

    def _add_holes_in_spiazzi(self, num_holes):
        count = 0
        attempts = 0
        while count < num_holes and attempts < 1000:
            attempts += 1
            # Random candidate for a hole (avoiding the immediate start/end)
            hx = self.rng.randint(2, self.width - 3)
            hy = self.rng.randint(2, self.height - 3)
            
            # Check if it's currently a path or a wall (we'll convert it to a "spiazzo")
            # We want to make sure the hole is at (hx, hy) and the 3x3 around it is path
            
            # Check if this area overlaps with start or end
            if abs(hx - 1) < 2 and abs(hy - 1) < 2: continue
            if abs(hx - (self.width - 2)) < 2 and abs(hy - (self.height - 2)) < 2: continue

            # Save state to revert if it breaks solvability
            backup = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    backup.append((hx + dx, hy + dy, self.grid[hy + dy][hx + dx]))
            
            # Create the 3x3 open area (spiazzo)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    self.grid[hy + dy][hx + dx] = 0
            
            # Place the hole
            self.grid[hy][hx] = 2
            
            # Check if the maze is still solvable
            if self.is_solvable():
                count += 1
            else:
                # Revert
                for x_coord, y_coord, val in backup:
                    self.grid[y_coord][x_coord] = val

    def is_solvable(self):
        # Find arrival
        goal = None
        start = (1, 1) # Default top-left
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 3:
                    goal = (x, y)
                elif self.grid[y][x] == 4:
                    start = (x, y)
            if goal and start != (1, 1): # If we found both and start changed
                 # we keep going to find goal or start if not found yet
                 pass
            
        if not goal: return False
        
        # BFS
        queue = deque([start])
        visited = {start}
        
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == goal:
                return True
                
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    # Path (0), Arrival (3) or Start (4) are traversable. Hole (2) and Wall (1) are not.
                    cell = self.grid[ny][nx]
                    if (nx, ny) not in visited and (cell == 0 or cell == 3 or cell == 4):
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False

    def get_bfs_distance_map(self):
        """
        Calculates a distance map from the goal (3) using BFS.
        Returns a 2D array where each cell contains the distance to the goal.
        """
        # Find arrival
        goal = None
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 3:
                    goal = (x, y)
                    break
            if goal: break
            
        if not goal: return None
        
        # Initialize distance map with a large value
        dist_map = np.full((self.height, self.width), 999, dtype=int)
        dist_map[goal[1]][goal[0]] = 0
        
        # BFS starting FROM the goal
        queue = deque([goal])
        visited = {goal}
        
        while queue:
            cx, cy = queue.popleft()
            current_dist = dist_map[cy][cx]
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    # Valid path: 0 (Path) or 1 (Start - wait, Start is just a pos, 1 is Wall).
                    # Actually logic: 0=Path, 1=Wall, 2=Hole, 3=Arrival.
                    # We can traverse 0 and 1 (if 1 was start, but 1 is WALL).
                    # The original code had `self.grid[ny][nx] == 1` which was WRONG (treating walls as walkable).
                    # We should only traverse 0 (Path) and 3 (Arrival) and potentially the start position if it was marked special, 
                    # but start is usually on a 0.
                    # Let's verify what 1 is. Class docstring says: 1 = Wall.
                    # So we MUST NOT traverse 1.
                    
                    if (nx, ny) not in visited:
                        cell_value = self.grid[ny][nx]
                        # We can walk on Path (0), Arrival (3), and Start (4).
                        # But standard walls (1) and Holes (2) are obstacles for walking.
                        
                        if cell_value == 0 or cell_value == 3 or cell_value == 4:
                            dist_map[ny][nx] = current_dist + 1
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                        
        return dist_map

    def get_action_mask_map(self):
        """
        Generates a (H, W, 9) boolean mask of valid actions.
        True = Valid Action, False = Invalid (Blocked by Wall).
        Action Map Indices correlate to:
        0: (-1, -1) -> Top-Right (Wait? Check logic below)
        1: (-1, 0)  -> Top
        2: (-1, 1)  -> Top-Left
        3: (0, -1)  -> Right
        4: (0, 0)   -> Neutral
        5: (0, 1)   -> Left
        6: (1, -1)  -> Bottom-Right
        7: (1, 0)   -> Bottom
        8: (1, 1)   -> Bottom-Left
        
        Note on Physics Mapping (Derived):
        d_tilt_x = -1 -> Force towards Top (Grid Y-)
        d_tilt_x = +1 -> Force towards Bottom (Grid Y+)
        d_tilt_y = +1 -> Force towards Left (Grid X-)
        d_tilt_y = -1 -> Force towards Right (Grid X+)
        """
        # Map actions to grid direction requirements
        # (d_tilt_x, d_tilt_y)
        # We need to block if the action pushes us into a wall.
        # Action Map from MarbleEnv:
        # 0: (-1, -1), 1: (-1, 0), 2: (-1, 1)
        # 3: (0, -1),  4: (0, 0),  5: (0, 1)
        # 6: (1, -1),  7: (1, 0),  8: (1, 1)
        
        # Mapping tilts to Grid Neighbors (dx, dy)
        # d_tilt_x=-1 -> Top (dy=-1)
        # d_tilt_x=+1 -> Bottom (dy=+1)
        # d_tilt_y=-1 -> Right (dx=+1)
        # d_tilt_y=+1 -> Left (dx=-1)
        
        # So action components:
        # (-1, -1) -> Top + Right (dy=-1, dx=+1)
        # (-1, 0)  -> Top         (dy=-1, dx=0)
        # (-1, 1)  -> Top + Left  (dy=-1, dx=-1)
        # (0, -1)  -> Right       (dy=0,  dx=+1)
        # (0, 0)   -> Neutral     (dy=0,  dx=0)
        # (0, 1)   -> Left        (dy=0,  dx=-1)
        # (1, -1)  -> Bottom + Right (dy=+1, dx=+1)
        # (1, 0)   -> Bottom      (dy=+1, dx=0)
        # (1, 1)   -> Bottom + Left (dy=+1, dx=-1)
        
        action_deltas = [
            (-1, -1), # 0: Top-Left  (d_tilt_y=-1 -> Left)
            (-1, 0),  # 1: Top
            (-1, 1),  # 2: Top-Right (d_tilt_y=+1 -> Right)
            (0, -1),  # 3: Left      (d_tilt_y=-1 -> Left)
            (0, 0),   # 4: Neutral
            (0, 1),   # 5: Right     (d_tilt_y=+1 -> Right)
            (1, -1),  # 6: Bottom-Left
            (1, 0),   # 7: Bottom
            (1, 1)    # 8: Bottom-Right
        ]
        # Physics Correction:
        # d_tilt_y = +1 (Index 5, 2, 8) -> Increases Pitch -> Rolls Right (X+). 
        # So dx should be +1.
        # d_tilt_y = -1 (Index 3, 0, 6) -> Decreases Pitch -> Rolls Left (X-).
        # So dx should be -1.
        
        # d_tilt_x = +1 (Index 7, 6, 8) -> Increases Roll -> Rolls Bottom (Y+).
        # So dy should be +1.
        # d_tilt_x = -1 (Index 1, 0, 2) -> Decreases Roll -> Rolls Top (Y-).
        # So dy should be -1.

        # WAIT. My manual derivation in comments earlier:
        # d_tilt_y = +1 -> Left (Grid X-). X- is Left?
        # Grid X0 is Left. X+ is Right. So X- is Left. Correct.
        # d_tilt_y = -1 -> Right (Grid X+).
        
        # Let's re-verify the table against MarbleEnv action_map and my derivation:
        # Env Map:    (-1, -1) (idx 0)
        # d_tilt_x = -1 (Top). d_tilt_y = -1 (Right).
        # Result: Top-Right.
        # My delta table above for 0: (-1, 1). 
        # (dy, dx). dy=-1 (Top). dx=1 (Right). Matches.
        
        # Env Map:    (-1, 1) (idx 2)
        # d_tilt_x = -1 (Top). d_tilt_y = +1 (Left).
        # Result: Top-Left.
        # My delta table for 2: (-1, -1).
        # (dy, dx). dy=-1 (Top). dx=-1 (Left). Matches.
        
        mask_map = np.ones((self.height, self.width, 9), dtype=bool)
        
        for y in range(self.height):
            for x in range(self.width):
                # If current cell is a Wall (1), allow EVERYTHING (as per user request).
                if self.grid[y][x] == 1:
                    continue
                
                # Check neighbors for blocking
                # We check the 4 cardinal directions. 
                # If a logical move has a component into a wall, we block it?
                # Or do we strictly block if the *immediate* neighbor in that direction is a wall?
                
                # Logic:
                # If Top is Wall -> Mask all actions with Top component.
                # If Right is Wall -> Mask all actions with Right component.
                
                top_blocked = (y > 0) and (self.grid[y-1][x] == 1)
                bottom_blocked = (y < self.height - 1) and (self.grid[y+1][x] == 1)
                left_blocked = (x > 0) and (self.grid[y][x-1] == 1)
                right_blocked = (x < self.width - 1) and (self.grid[y][x+1] == 1)
                
                # Border checks (Treat boundaries as walls)
                if y == 0: top_blocked = True
                if y == self.height - 1: bottom_blocked = True
                if x == 0: left_blocked = True
                if x == self.width - 1: right_blocked = True
                
                for a_idx, (dy, dx) in enumerate(action_deltas):
                    is_blocked = False
                    
                    if dy == -1 and top_blocked: is_blocked = True
                    if dy == 1 and bottom_blocked: is_blocked = True
                    if dx == -1 and left_blocked: is_blocked = True
                    if dx == 1 and right_blocked: is_blocked = True
                    
                    if is_blocked:
                        mask_map[y, x, a_idx] = False
                        
        return mask_map

    def get_random_valid_cell(self):
        valid_cells = []
        for y in range(self.height):
            for x in range(self.width):
                # 0 is Path, we avoid the arrival zone (3) for spawning if possible
                if self.grid[y][x] == 0:
                    # Map grid coordinates to the same coordinate system used in setup_world (cx, cy)
                    valid_cells.append((x, y))
        
        if not valid_cells:
            return (1, 1) # Default start
            
        return self.rng.choice(valid_cells)

if __name__ == "__main__":
    # Quick test
    gen = MazeGenerator(15, 15, 100)
    grid = gen.generate()
    for row in grid:
        symbols = {0: " ", 1: "#", 2: "O", 3: "X", 4: "S"}
        print("".join(symbols[c] for c in row))
    print("Solvable:", gen.is_solvable())
    print("Distance Map: ")
    print(gen.get_bfs_distance_map())

