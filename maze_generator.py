import random
from collections import deque

class MazeGenerator:
    """
    Generates a maze using DFS and places holes in 'spiazzi' (open areas).
    0 = Path
    1 = Wall
    2 = Hole
    3 = Arrival (Goal)
    """
    def __init__(self, width, height):
        # Ensure dimensions are odd for DFS wall carving
        self.width = width if width % 2 != 0 else width + 1
        self.height = height if height % 2 != 0 else height + 1
        self.grid = [[1 for _ in range(self.width)] for _ in range(self.height)]
        
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
        random.shuffle(directions)
        
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
            hx = random.randint(2, self.width - 3)
            hy = random.randint(2, self.height - 3)
            
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

if __name__ == "__main__":
    # Quick test
    gen = MazeGenerator(15, 15)
    grid = gen.generate()
    for row in grid:
        symbols = {0: " ", 1: "#", 2: "O", 3: "X"}
        print("".join(symbols[c] for c in row))
    print("Solvable:", gen.is_solvable())
