import open3d as o3d
import numpy as np
import random

# Function to compute A3 descriptor (angle between 3 random vertices)
def compute_a3(vertices, num_samples, bins):
    angles = []
    for _ in range(num_samples):
        v1, v2, v3 = random.sample(list(vertices), 3)
        vec1 = v2 - v1
        vec2 = v3 - v1
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        angle = np.arccos(np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0))
        angles.append(np.degrees(angle))
    hist, _ = np.histogram(angles, bins=bins, range=(0, 180))
    return hist

# Function to compute D1 descriptor (distance between barycenter and random vertex) with bins
def compute_d1(vertices, num_samples, bins):
    barycenter = np.mean(vertices, axis=0)
    distances = []
    for _ in range(num_samples):
        v = random.choice(vertices)
        dist = np.linalg.norm(v - barycenter)
        distances.append(dist)
    hist, _ = np.histogram(distances, bins=bins)
    return hist

# Function to compute D2 descriptor (distance between 2 random vertices) with bins
def compute_d2(vertices, num_samples, bins):
    distances = []
    for _ in range(num_samples):
        v1, v2 = random.sample(list(vertices), 2)
        dist = np.linalg.norm(v1 - v2)
        distances.append(dist)
    hist, _ = np.histogram(distances, bins=bins)
    return hist

# Function to compute D3 descriptor (square root of area of triangle given by 3 random vertices) with bins
def compute_d3(vertices, num_samples, bins):
    areas = []
    for _ in range(num_samples):
        v1, v2, v3 = random.sample(list(vertices), 3)
        a = np.linalg.norm(v1 - v2)
        b = np.linalg.norm(v2 - v3)
        c = np.linalg.norm(v3 - v1)
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        areas.append(area)
    hist, _ = np.histogram(np.sqrt(areas), bins=bins)
    return hist

# Function to compute D4 descriptor (cube root of volume of tetrahedron formed by 4 random vertices) with bins
def compute_d4(vertices, num_samples, bins):
    volumes = []
    for _ in range(num_samples):
        v1, v2, v3, v4 = random.sample(list(vertices), 4)
        mat = np.vstack((v2 - v1, v3 - v1, v4 - v1)).T
        volume = np.abs(np.linalg.det(mat)) / 6.0
        volumes.append(volume)
    hist, _ = np.histogram(np.cbrt(volumes), bins=bins)
    return hist

# Function to compute all descriptors for a shape
def compute_descriptors_for_shape(mesh, shape_name, num_samples, bins):
    vertices = np.asarray(mesh.vertices)

    # Compute all the descriptors
    a3_hist = compute_a3(vertices, num_samples, bins)
    d1_hist = compute_d1(vertices, num_samples, bins)
    d2_hist = compute_d2(vertices, num_samples, bins)
    d3_hist = compute_d3(vertices, num_samples, bins)
    d4_hist = compute_d4(vertices, num_samples, bins)

    # Concatenate all histograms into a single descriptor vector
    shape_descriptor = np.concatenate([a3_hist, d1_hist, d2_hist, d3_hist, d4_hist])

    # Print the results for verification
    print(f"Shape: {shape_name}")
    print(f"Shape Descriptor (Concatenated Histograms): {shape_descriptor}\n")

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
    # Set the number of samples and bins
    num_samples = 1000  # You can modify this easily
    bins = 10  # You can modify this easily

    # Create simple shapes
    cube = create_cube()
    sphere = create_sphere()
    cylinder = create_cylinder()

    # Check descriptors for simple shapes
    compute_descriptors_for_shape(cube, "Cube", num_samples, bins)
    compute_descriptors_for_shape(sphere, "Sphere", num_samples, bins)
    compute_descriptors_for_shape(cylinder, "Cylinder", num_samples, bins)
