import random

import open3d as o3d
import numpy as np
import os
import csv
from multiprocessing import Pool

# Number of samples for each descriptor
NUM_SAMPLES = 5000

# A3: Compute angles between three random vertices
def compute_a3(vertices, num_samples):
    angles = []
    for _ in range(num_samples):
        a, b, c = vertices[np.random.choice(len(vertices), 3, replace=False)]
        vec1, vec2 = b - a, c - a
        angle = np.degrees(np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0)))
        angles.append(angle)
    return angles

# D1: Compute distance from barycenter to a random vertex
def random_point_in_triangle(v1, v2, v3):
    # Generate random barycentric coordinates
    r1, r2 = np.random.rand(2)
    # Adjust coordinates to maintain uniform distribution within the triangle
    if r1 + r2 > 1:
        r1, r2 = 1 - r1, 1 - r2
    # Compute the point in the triangle
    return (1 - r1 - r2) * v1 + r1 * v2 + r2 * v3

def compute_d1(vertices, num_samples):
    barycenter = np.mean(vertices, axis=0)
    distances = []
    for _ in range(num_samples):
        # Choose 3 random vertices to define a triangle
        v1, v2, v3 = random.sample(list(vertices), 3)
        # Sample a random point within the triangle formed by v1, v2, and v3
        point = random_point_in_triangle(v1, v2, v3)
        # Calculate the distance from the sampled point to the barycenter
        distances.append(np.linalg.norm(point - barycenter))
    return distances

# D2: Compute distance between two random vertices
def compute_d2(vertices, num_samples):
    distances = []
    for _ in range(num_samples):
        a, b = vertices[np.random.choice(len(vertices), 2, replace=False)]
        distances.append(np.linalg.norm(a - b))
    return distances

# D3: Compute area of triangles formed by three random vertices
def compute_d3(vertices, num_samples):
    areas = []
    for _ in range(num_samples):
        a, b, c = vertices[np.random.choice(len(vertices), 3, replace=False)]
        side_a, side_b, side_c = np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)
        s = (side_a + side_b + side_c) / 2
        area = np.sqrt(max(s * (s - side_a) * (s - side_b) * (s - side_c), 0))
        areas.append(area)
    return areas

# D4: Compute volume of tetrahedrons formed by four random vertices
def compute_d4(vertices, num_samples):
    volumes = []
    for _ in range(num_samples):
        a, b, c, d = vertices[np.random.choice(len(vertices), 4, replace=False)]
        volume = np.abs(np.dot(a - d, np.cross(b - d, c - d))) / 6
        volumes.append(volume)
    return volumes

# Compute descriptors and save to CSV
def compute_shape_descriptors(args):
    model_path, category, output_csv, num_samples = args
    model_name = os.path.basename(model_path)
    print(f"Processing {model_name} in {category}...")

    mesh = o3d.io.read_triangle_mesh(model_path)
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)

    # Compute each descriptor
    a3_data = compute_a3(vertices, num_samples)
    d1_data = compute_d1(vertices, num_samples)
    d2_data = compute_d2(vertices, num_samples)
    d3_data = compute_d3(vertices, num_samples)
    d4_data = compute_d4(vertices, num_samples)

    # Save to CSV
    with open(output_csv, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for i in range(num_samples):
            writer.writerow([category, model_name, a3_data[i], d1_data[i], d2_data[i], d3_data[i], d4_data[i]])

# Process database with multiprocessing
def process_database(database_path, num_samples=5000, output_csv='shape_descriptors.csv'):
    # Create CSV with header
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['category', 'model', 'A3', 'D1', 'D2', 'D3', 'D4'])

    # Prepare arguments for multiprocessing
    args_list = []
    for category in os.listdir(database_path):
        category_path = os.path.join(database_path, category)
        if os.path.isdir(category_path):
            for model in os.listdir(category_path):
                if model.endswith('.obj'):
                    model_path = os.path.join(category_path, model)
                    args_list.append((model_path, category, output_csv, num_samples))

    # Use multiprocessing Pool to process models
    with Pool(processes=1) as pool:
        pool.map(compute_shape_descriptors, args_list)

if __name__ == "__main__":
    # Define path to database and output parameters
    database_path = 'ShapeDatabase_INFOMR-master'
    num_samples = 5000  # Higher sample count for detailed models
    output_csv = 'shape_descriptors.csv'

    # Process database and save descriptors
    process_database(database_path, num_samples, output_csv)
    print(f"Processing complete. Results saved to {output_csv}")