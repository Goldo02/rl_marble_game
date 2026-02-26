import numpy as np
from maze_generator import MazeGenerator

def debug_bfs():
    # Mirror parameters from train.py
    seed = 100
    width = 15
    height = 15
    
    gen = MazeGenerator(width, height, seed=seed)
    grid = gen.generate()
    bfs_map = gen.get_bfs_distance_map()
    
    print("MAZE MAP:")
    symbols = {0: " ", 1: "#", 2: "O", 3: "X", 4: "S"}
    for row in grid:
        print("".join(symbols[c] for c in row))
        
    print("\nBFS DIRECTIONS:")
    for y in range(len(grid)):
        line = ""
        for x in range(len(grid[0])):
            if grid[y][x] == 1:
                line += "#"
                continue
            if grid[y][x] == 3:
                line += "X"
                continue
                
            # Logic from MarbleEnv._get_obs (simplified)
            min_bfs = bfs_map[y][x]
            best_dir = "?"
            neighbors = [
                (0, -1, "N"), (0, 1, "S"), (1, 0, "E"), (-1, 0, "W")
            ]
            for dx, dy, name in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if bfs_map[ny][nx] < min_bfs:
                        min_bfs = bfs_map[ny][nx]
                        best_dir = name
            
            # Map N/S/E/W to arrows
            arrows = {"N": "↑", "S": "↓", "E": "→", "W": "←", "?": "·"}
            line += arrows[best_dir]
        print(line)

if __name__ == "__main__":
    debug_bfs()
