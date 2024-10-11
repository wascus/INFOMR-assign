import open3d as o3d
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Function to compute A3 descriptor (angle between 3 random vertices)
def compute_a3(vertices, num_samples):
    angles = []
    for _ in range(num_samples):
        v1, v2, v3 = random.sample(list(vertices), 3)
        vec1 = v2 - v1
        vec2 = v3 - v1
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        angle = np.arccos(np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0))
        angles.append(np.degrees(angle))
    return angles

# Function to compute D1 descriptor (distance between barycenter and random vertex)
def compute_d1(vertices, num_samples):
    barycenter = np.mean(vertices, axis=0)
    distances = []
    for _ in range(num_samples):
        v = random.choice(vertices)
        dist = np.linalg.norm(v - barycenter)
        distances.append(dist)
    return distances

# Function to compute D2 descriptor (distance between 2 random vertices)
def compute_d2(vertices, num_samples):
    distances = []
    for _ in range(num_samples):
        v1, v2 = random.sample(list(vertices), 2)
        dist = np.linalg.norm(v1 - v2)
        distances.append(dist)
    return distances

# Function to compute D3 descriptor (square root of area of triangle given by 3 random vertices)
def compute_d3(vertices, num_samples):
    areas = []
    for _ in range(num_samples):
        v1, v2, v3 = random.sample(list(vertices), 3)
        a = np.linalg.norm(v1 - v2)
        b = np.linalg.norm(v2 - v3)
        c = np.linalg.norm(v3 - v1)
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        areas.append(area)
    return areas

# Function to compute D4 descriptor (cube root of volume of tetrahedron formed by 4 random vertices)
def compute_d4(vertices, num_samples):
    volumes = []
    for _ in range(num_samples):
        v1, v2, v3, v4 = random.sample(list(vertices), 4)
        mat = np.vstack((v2 - v1, v3 - v1, v4 - v1)).T
        volume = np.abs(np.linalg.det(mat)) / 6.0
        volumes.append(volume)
    return volumes

# Function to plot the raw data as a histogram or KDE plot
def plot_data(data, title):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data, fill=True)  # KDE plot for smooth data visualization
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

# Function to compute all descriptors for a shape and visualize them
def compute_descriptors_for_shape(mesh, shape_name, num_samples):
    vertices = np.asarray(mesh.vertices)

    # Compute all the descriptors using the raw data
    a3_data = compute_a3(vertices, num_samples)
    d1_data = compute_d1(vertices, num_samples)
    d2_data = compute_d2(vertices, num_samples)
    d3_data = compute_d3(vertices, num_samples)
    d4_data = compute_d4(vertices, num_samples)

    # Print the raw data for debugging or inspection
    print(f"Shape: {shape_name}")
    print(f"A3 Descriptor (Raw Data): {a3_data[:10]}...")  # Printing only the first 10 values
    print(f"D1 Descriptor (Raw Data): {d1_data[:10]}...")
    print(f"D2 Descriptor (Raw Data): {d2_data[:10]}...")
    print(f"D3 Descriptor (Raw Data): {d3_data[:10]}...")
    print(f"D4 Descriptor (Raw Data): {d4_data[:10]}...\n")

    # Plot the raw data using KDE or raw histograms
    plot_data(a3_data, f"{shape_name}: A3 Descriptor (Angle Distribution)")
    plot_data(d1_data, f"{shape_name}: D1 Descriptor (Distance to Barycenter)")
    plot_data(d2_data, f"{shape_name}: D2 Descriptor (Distance Between Vertices)")
    plot_data(d3_data, f"{shape_name}: D3 Descriptor (Triangle Area Distribution)")
    plot_data(d4_data, f"{shape_name}: D4 Descriptor (Tetrahedron Volume Distribution)")

# Generate a high-resolution cube by subdividing the faces
def create_high_res_cube():
    cube = o3d.geometry.TriangleMesh.create_box(width=5.0, height=5.0, depth=5.0)
    cube.compute_vertex_normals()
    # Subdivide the cube to increase the vertex count
    cube = cube.subdivide_midpoint(number_of_iterations=3)  # 3 iterations of subdivision for higher resolution
    return cube

# Generate a high-resolution sphere by increasing its resolution
def create_high_res_sphere():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3.0, resolution=50)  # Higher resolution sphere
    sphere.compute_vertex_normals()
    return sphere

# Generate a high-resolution cylinder by increasing its resolution
def create_high_res_cylinder():
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=2.0, height=8.0, resolution=50, split=20)
    cylinder.compute_vertex_normals()
    return cylinder

# Main function to generate shapes and check their descriptors
if __name__ == "__main__":
    # Set the number of samples
    num_samples = 10000  # You can modify this easily

    # Create high-resolution shapes
    cube = create_high_res_cube()
    sphere = create_high_res_sphere()
    cylinder = create_high_res_cylinder()

    # Check descriptors for simple shapes and plot them
    compute_descriptors_for_shape(cube, "High-Resolution Cube", num_samples)
    compute_descriptors_for_shape(sphere, "High-Resolution Sphere", num_samples)
    compute_descriptors_for_shape(cylinder, "High-Resolution Cylinder", num_samples)
