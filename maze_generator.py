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
        
        # 3. Add Holes in "Spiazzi"
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
        start = (1, 1)
        # Find arrival
        goal = None
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 3:
                    goal = (x, y)
                    break
            if goal: break
            
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
                    # Path (0) or Arrival (3) are traversable. Hole (2) and Wall (1) are not.
                    if (nx, ny) not in visited and (self.grid[ny][nx] == 0 or self.grid[ny][nx] == 3):
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
                        # We can walk on Path (0), Arrival (3), and maybe if Start was marked separately.
                        # But standard walls (1) and Holes (2) are obstacles for walking.
                        # Note regarding Holes (2): Physics-wise, you fall. So BFS should probably avoid them 
                        # OR treat them as valid paths but high risk? 
                        # If we want the agent to learn to jump over them, maybe? 
                        # But for now, let's treat them as non-traversable for the *distance map* 
                        # so the gradient guides around them if possible. 
                        # If a hole is blocking the ONLY path, then BFS will fail, which is correct (unsolvable).
                        # But `_add_holes_in_spiazzi` checks solvability, likely assuming 2 is NOT traversable.
                        # Let's check `is_solvable`: `if (self.grid[ny][nx] == 0 or self.grid[ny][nx] == 3):`
                        # So `is_solvable` treats 0 and 3 as walkable. 2 and 1 are NOT.
                        
                        if cell_value == 0 or cell_value == 3:
                            dist_map[ny][nx] = current_dist + 1
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                        
        return dist_map

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
    gen = MazeGenerator(15, 15)
    grid = gen.generate()
    for row in grid:
        symbols = {0: " ", 1: "#", 2: "O", 3: "X"}
        print("".join(symbols[c] for c in row))
    print("Solvable:", gen.is_solvable())
