import os

class ObjBuilder:
    """
    Helper to generate .obj files for PyBullet.
    Focuses on simple boxes for floor and walls.
    """
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.vertex_offset = 1 # OBJ is 1-indexed

    def add_box(self, center, half_extents):
        """
        Adds a box volume to the mesh.
        center: (x, y, z)
        half_extents: (hx, hy, hz)
        """
        cx, cy, cz = center
        hx, hy, hz = half_extents

        # 8 corners
        # Bottom
        v1 = (cx - hx, cy - hy, cz - hz)
        v2 = (cx + hx, cy - hy, cz - hz)
        v3 = (cx + hx, cy + hy, cz - hz)
        v4 = (cx - hx, cy + hy, cz - hz)
        # Top
        v5 = (cx - hx, cy - hy, cz + hz)
        v6 = (cx + hx, cy - hy, cz + hz)
        v7 = (cx + hx, cy + hy, cz + hz)
        v8 = (cx - hx, cy + hy, cz + hz)

        new_verts = [v1, v2, v3, v4, v5, v6, v7, v8]
        self.vertices.extend(new_verts)

        o = self.vertex_offset
        
        def add_quad(i1, i2, i3, i4):
            # i1-i2-i3 and i1-i3-i4
            self.faces.append((o+i1, o+i2, o+i3))
            self.faces.append((o+i1, o+i3, o+i4))

        # Bottom
        add_quad(3, 2, 1, 0)
        # Top
        add_quad(4, 5, 6, 7)
        # Side Front (y-)
        add_quad(0, 1, 5, 4)
        # Side Right (x+)
        add_quad(1, 2, 6, 5)
        # Side Back (y+)
        add_quad(2, 3, 7, 6)
        # Side Left (x-)
        add_quad(3, 0, 4, 7)

        self.vertex_offset += 8

    def write_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write("# Marble Game Mesh\n")
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in self.faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")
        return os.path.abspath(filename)
