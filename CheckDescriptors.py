import open3d as o3d
import numpy as np
import random

# Function to compute A3 descriptor (angle between 3 random vertices)
def compute_a3(vertices, num_samples=10000, bins=10):
    angles = []
    for _ in range(num_samples):
        v1, v2, v3 = random.sample(list(vertices), 3)
        vec1 = v2 - v1
        vec2 = v3 - v1
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        angle = np.arccos(np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0))
        angles.append(np.degrees(angle))
    return np.mean(angles)

# Function to compute D1 descriptor (distance between barycenter and random vertex)
def compute_d1(vertices, num_samples=10000):
    barycenter = np.mean(vertices, axis=0)
    distances = []
    for _ in range(num_samples):
        v = random.choice(vertices)
        dist = np.linalg.norm(v - barycenter)
        distances.append(dist)
    return np.mean(distances)

# Function to compute D2 descriptor (distance between 2 random vertices)
def compute_d2(vertices, num_samples=10000):
    distances = []
    for _ in range(num_samples):
        v1, v2 = random.sample(list(vertices), 2)
        dist = np.linalg.norm(v1 - v2)
        distances.append(dist)
    return np.mean(distances)

# Function to compute D3 descriptor (square root of area of triangle given by 3 random vertices)
def compute_d3(vertices, num_samples=10000):
    areas = []
    for _ in range(num_samples):
        v1, v2, v3 = random.sample(list(vertices), 3)
        a = np.linalg.norm(v1 - v2)
        b = np.linalg.norm(v2 - v3)
        c = np.linalg.norm(v3 - v1)
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        areas.append(area)
    return np.mean(np.sqrt(areas))

# Function to compute D4 descriptor (cube root of volume of tetrahedron formed by 4 random vertices)
def compute_d4(vertices, num_samples=10000):
    volumes = []
    for _ in range(num_samples):
        v1, v2, v3, v4 = random.sample(list(vertices), 4)
        mat = np.vstack((v2 - v1, v3 - v1, v4 - v1)).T
        volume = np.abs(np.linalg.det(mat)) / 6.0
        volumes.append(volume)
    return np.mean(np.cbrt(volumes))

# Function to compute all descriptors for a shape
def compute_descriptors_for_shape(mesh, shape_name):
    vertices = np.asarray(mesh.vertices)

    # Compute all the descriptors
    a3 = compute_a3(vertices)
    d1 = compute_d1(vertices)
    d2 = compute_d2(vertices)
    d3 = compute_d3(vertices)
    d4 = compute_d4(vertices)

    # Print the results for verification
    print(f"Shape: {shape_name}")
    print(f"A3 (Average angle between 3 vertices): {a3:.4f} degrees")
    print(f"D1 (Avg. distance between barycenter and random vertex): {d1:.4f}")
    print(f"D2 (Avg. distance between 2 random vertices): {d2:.4f}")
    print(f"D3 (Avg. sqrt of area of triangle): {d3:.4f}")
    print(f"D4 (Avg. cube root of volume of tetrahedron): {d4:.4f}\n")

# Generate a cube
def create_cube():
    cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    cube.compute_vertex_normals()
    return cube

# Generate a sphere
def create_sphere():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere.compute_vertex_normals()
    return sphere

# Generate a cylinder
def create_cylinder():
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.5, height=3.0)
    cylinder.compute_vertex_normals()
    return cylinder

# Main function to generate shapes and check their descriptors
if __name__ == "__main__":
    # Create simple shapes
    cube = create_cube()
    sphere = create_sphere()
    cylinder = create_cylinder()

    # Check descriptors for simple shapes
    compute_descriptors_for_shape(cube, "Cube")
    compute_descriptors_for_shape(sphere, "Sphere")
    compute_descriptors_for_shape(cylinder, "Cylinder")

    # Visualize the shapes (optional)
    o3d.visualization.draw_geometries([cube], window_name="Cube")
    o3d.visualization.draw_geometries([sphere], window_name="Sphere")
    o3d.visualization.draw_geometries([cylinder], window_name="Cylinder")
