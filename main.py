import pybullet as p
import pybullet_data
import time
import math
import numpy as np
from maze_generator import MazeGenerator
from mesh_utils import ObjBuilder

class MarbleGame:
    def __init__(self, gui=True):
        self.gui = gui
        if self.gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Simulation parameters
        self.time_step = 1/240
        p.setTimeStep(self.time_step)
        
        # Board settings
        self.square_size = 2.0  # half-extent
        self.cell_size = 0.2
        self.wall_height = 0.3
        self.floor_thickness = 0.2 # Thick floor to avoid tunneling
        
        # Maze dimensions
        self.maze_width = int((2 * self.square_size) / self.cell_size)
        self.maze_height = int((2 * self.square_size) / self.cell_size)
        if self.maze_width % 2 == 0: self.maze_width -= 1
        if self.maze_height % 2 == 0: self.maze_height -= 1
        
        # Control parameters
        self.max_tilt = math.radians(10)
        self.min_tilt = -math.radians(10)
        self.tilt_speed = 0.002 # Very gradual for precision
        
        self.current_tilt_x = 0
        self.current_tilt_y = 0
        self.stabilize_counter = 100 # Frames to stay flat at start
        
        self.setup_world()
        self.reset() 
        
    def setup_world(self):
        # 1. Generate Maze
        self.maze_gen = MazeGenerator(self.maze_width, self.maze_height)
        self.grid = self.maze_gen.generate()
        
        # 2. Build Mesh
        builder = ObjBuilder()
        offset_x = -(self.maze_width * self.cell_size) / 2
        offset_y = -(self.maze_height * self.cell_size) / 2
        
        self.arrival_pos = None
        self.start_pos = None
        
        # Visual/Collision Arrays for MultiBody
        col_shapes = []
        col_extents = []
        col_positions = []
        
        vis_shapes = []
        vis_extents = []
        vis_positions = []
        vis_colors = []

        for y in range(self.maze_height):
            for x in range(self.maze_width):
                cell = self.grid[y][x]
                cx = offset_x + (x * self.cell_size) + (self.cell_size / 2)
                cy = offset_y + (y * self.cell_size) + (self.cell_size / 2)
                
                # Floor part (common for path, wall base, and arrival)
                if cell != 2: # No floor in holes
                    # Mesh Builder (for visual)
                    builder.add_box((cx, cy, -self.floor_thickness/2), (self.cell_size/2, self.cell_size/2, self.floor_thickness/2))
                    
                    # Physics (Collision)
                    col_shapes.append(p.GEOM_BOX)
                    col_extents.append([self.cell_size/2, self.cell_size/2, self.floor_thickness/2])
                    col_positions.append([cx, cy, -self.floor_thickness/2])
                
                if cell == 1: # Wall
                    builder.add_box((cx, cy, self.wall_height/2), (self.cell_size/2, self.cell_size/2, self.wall_height/2))
                    col_shapes.append(p.GEOM_BOX)
                    col_extents.append([self.cell_size/2, self.cell_size/2, self.wall_height/2])
                    col_positions.append([cx, cy, self.wall_height/2])
                
                elif cell == 3: # Arrival
                    self.arrival_pos = np.array([cx, cy, 0.02])
                    # Visual marker (Magenta tile, high contrast)
                    vis_shapes.append(p.GEOM_BOX)
                    vis_extents.append([self.cell_size/2, self.cell_size/2, 0.02])
                    vis_positions.append([cx, cy, 0.02])
                    vis_colors.append([1, 0, 1, 1])
                
                if x == 1 and y == 1:
                    self.start_pos = [cx, cy, 0.5] # Higher spawn to land safely

        # Borders
        board_h = self.square_size
        th = 0.05
        # Add side borders to the mesh
        builder.add_box((0, board_h + th/2, self.wall_height/2), (board_h + th, th/2, self.wall_height/2 + self.floor_thickness/2))
        builder.add_box((0, -board_h - th/2, self.wall_height/2), (board_h + th, th/2, self.wall_height/2 + self.floor_thickness/2))
        builder.add_box((board_h + th/2, 0, self.wall_height/2), (th/2, board_h, self.wall_height/2 + self.floor_thickness/2))
        builder.add_box((-board_h - th/2, 0, self.wall_height/2), (th/2, board_h, self.wall_height/2 + self.floor_thickness/2))

        # Collision for borders
        for pos, ext in [
            ([0, board_h + th/2, self.wall_height/2], [board_h + th, th/2, self.wall_height/2 + self.floor_thickness/2]),
            ([0, -board_h - th/2, self.wall_height/2], [board_h + th, th/2, self.wall_height/2 + self.floor_thickness/2]),
            ([board_h + th/2, 0, self.wall_height/2], [th/2, board_h, self.wall_height/2 + self.floor_thickness/2]),
            ([-board_h - th/2, 0, self.wall_height/2], [th/2, board_h, self.wall_height/2 + self.floor_thickness/2])
        ]:
            col_shapes.append(p.GEOM_BOX)
            col_extents.append(ext)
            col_positions.append(pos)

        obj_path = builder.write_to_file("board.obj")
        
        maze_vis = p.createVisualShape(p.GEOM_MESH, fileName=obj_path, rgbaColor=[0.8, 0.7, 0.5, 1])
        
        # Physics Construction (Using Concave Trimesh for high-fidelity labyrinth)
        # Note: flags=p.GEOM_FORCE_CONCAVE_TRIMESH is CRITICAL for non-convex static meshes
        maze_col = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        
        self.board_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=maze_col,
            baseVisualShapeIndex=maze_vis,
            basePosition=[0, 0, 0]
        )
        
        self.local_arrival_pos = self.arrival_pos.copy() 
        
        # 2.5 Destination Marker (Separate Body for guaranteed color/visibility)
        marker_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.cell_size/2, self.cell_size/2, 0.01], rgbaColor=[1, 0, 1, 1])
        self.marker_id = p.createMultiBody(
            baseMass=0.01,
            baseVisualShapeIndex=marker_vis,
            basePosition=self.arrival_pos
        )
        # Fix marker to board
        p.createConstraint(self.board_id, -1, self.marker_id, -1, p.JOINT_FIXED, [0,0,0], self.arrival_pos, [0,0,0])
        # 3. Ball (Marble)
        radius = 0.08
        ball_col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        ball_vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0.9, 0.1, 0.1, 1])
        self.ball_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=ball_col,
            baseVisualShapeIndex=ball_vis,
            basePosition=self.start_pos
        )
        
        # Physics Tweaks
        p.changeDynamics(self.ball_id, -1, 
                         lateralFriction=0.1, 
                         rollingFriction=0.002, 
                         spinningFriction=0.002, 
                         restitution=0.2, 
                         linearDamping=0.01,
                         angularDamping=0.01,
                         ccdSweptSphereRadius=radius)
        
        # Disable sleeping so the ball always responds to board tilt
        p.changeDynamics(self.ball_id, -1, activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)
        
        p.changeDynamics(self.board_id, -1, lateralFriction=0.1, restitution=0.2)

    def step(self):
        keys = p.getKeyboardEvents()
        
        # Determine targets
        if self.stabilize_counter > 0:
            target_x = 0
            target_y = 0
            self.stabilize_counter -= 1
        else:
            target_x = self.min_tilt
            target_y = self.min_tilt
        
        # Up Arrow for X-axis tilt positive (+10)
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            target_x = self.max_tilt
            
        # Right Arrow for Y-axis tilt positive (+10)
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            target_y = self.max_tilt
            
        # Smooth interpolation
        if self.current_tilt_x < target_x:
            self.current_tilt_x = min(target_x, self.current_tilt_x + self.tilt_speed)
        elif self.current_tilt_x > target_x:
            self.current_tilt_x = max(target_x, self.current_tilt_x - self.tilt_speed)
            
        if self.current_tilt_y < target_y:
            self.current_tilt_y = min(target_y, self.current_tilt_y + self.tilt_speed)
        elif self.current_tilt_y > target_y:
            self.current_tilt_y = max(target_y, self.current_tilt_y - self.tilt_speed)
            
        # Apply orientation to board
        # In PyBullet, Euler are [roll, pitch, yaw]. 
        # We'll map tilt_x to Pitch and tilt_y to Roll.
        orn = p.getQuaternionFromEuler([self.current_tilt_y, self.current_tilt_x, 0])
        p.resetBasePositionAndOrientation(self.board_id, [0, 0, 0], orn)
        
        p.stepSimulation()
        
        # Logic checks
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        
        # Convert local arrival to world space
        pos_board, orn_board = p.getBasePositionAndOrientation(self.board_id)
        rot_mat = p.getMatrixFromQuaternion(orn_board)
        rot_mat = np.array(rot_mat).reshape(3, 3)
        world_arrival = rot_mat @ self.local_arrival_pos + np.array(pos_board)
        
        # Fall check (Hole)
        # Check if ball is below the board center and far from the start
        if ball_pos[2] < -0.5:
            print("GAME OVER - Fell into a hole!")
            time.sleep(1)
            self.reset()
            
        # Win check
        dist = np.linalg.norm(np.array(ball_pos) - world_arrival)
        # Check if 3D distance is small enough (ball radius is 0.08, cell is 0.2)
        if dist < (self.cell_size * 0.8):
            print("YOU WIN! - Reached the Goal!")
            time.sleep(2)
            self.reset()
            
        if self.gui:
            time.sleep(self.time_step)

    def reset(self):
        p.resetBasePositionAndOrientation(self.ball_id, self.start_pos, [0,0,0,1])
        p.resetBaseVelocity(self.ball_id, [0,0,0], [0,0,0])
        self.current_tilt_x = self.min_tilt
        self.current_tilt_y = self.min_tilt
        p.resetBasePositionAndOrientation(self.board_id, [0,0,0], p.getQuaternionFromEuler([self.min_tilt, self.min_tilt, 0]))

if __name__ == "__main__":
    game = MarbleGame()
    p.resetDebugVisualizerCamera(cameraDistance=5.5, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0,0,0])
    
    while True:
        game.step()
