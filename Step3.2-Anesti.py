import open3d as o3d
import numpy as np
import random
import os
import csv

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

# Function to compute all shape descriptors for a single file
def compute_shape_descriptors(file_path, num_samples=1000, bins=10):
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)

    # Compute all histograms
    a3_histogram = compute_a3(vertices, num_samples, bins)
    d1_histogram = compute_d1(vertices, num_samples, bins)
    d2_histogram = compute_d2(vertices, num_samples, bins)
    d3_histogram = compute_d3(vertices, num_samples, bins)
    d4_histogram = compute_d4(vertices, num_samples, bins)

    # Concatenate all histograms into a single descriptor vector
    shape_descriptor = np.concatenate([a3_histogram, d1_histogram, d2_histogram, d3_histogram, d4_histogram])

    return shape_descriptor

# Function to iterate over all subfolders and compute descriptors for each model
def process_database(database_path, num_samples=1000, bins=10, output_csv='shape_descriptors.csv'):
    # Open CSV file for writing
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        header = ['category', 'model'] + [f'feature_{i+1}' for i in range(bins * 5)]
        writer.writerow(header)

        for category in os.listdir(database_path):
            category_path = os.path.join(database_path, category)
            if os.path.isdir(category_path):  # Check if it's a folder
                for model in os.listdir(category_path):
                    if model.endswith('.obj'):  # Check if it's an .obj file
                        model_path = os.path.join(category_path, model)
                        descriptor = compute_shape_descriptors(model_path, num_samples, bins)
                        # Write the category, model name, and descriptor to the CSV
                        writer.writerow([category, model] + list(descriptor))
                        print(f"Processed {model} in {category}")

# Main block to run the script
if __name__ == "__main__":
    # Define the path to your database
    database_path = 'ShapeDatabase_INFOMR-master'

    # Set the number of samples and bins for histograms
    num_samples = 1000
    bins = 10

    # Process the entire database and save results to a CSV file
    output_csv = 'shape_descriptors.csv'
    process_database(database_path, num_samples, bins, output_csv)

    print(f"Processing complete. Results saved to {output_csv}")
