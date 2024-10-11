import open3d as o3d
import numpy as np
import random
import os
import csv

# Function to compute a random point inside a triangle using barycentric coordinates
def random_point_in_triangle(v1, v2, v3):
    r1, r2 = random.random(), random.random()
    if r1 + r2 > 1:
        r1, r2 = 1 - r1, 1 - r2
    return (1 - r1 - r2) * v1 + r1 * v2 + r2 * v3

# Function to compute A3 descriptor (angle between 3 random points inside random triangles)
def compute_a3(vertices, num_samples):
    angles = []
    for _ in range(num_samples):
        v1, v2, v3 = random.sample(list(vertices), 3)
        p1 = random_point_in_triangle(v1, v2, v3)
        p2 = random_point_in_triangle(*random.sample(list(vertices), 3))
        p3 = random_point_in_triangle(*random.sample(list(vertices), 3))

        vec1 = p2 - p1
        vec2 = p3 - p1
        angle = np.degrees(np.arccos(np.clip(np.dot(vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)), -1.0, 1.0)))
        angles.append(angle)
    return angles

# Function to compute D1 descriptor (distance between barycenter and random point inside a triangle)
def compute_d1(vertices, num_samples):
    barycenter = np.mean(vertices, axis=0)
    distances = []
    for _ in range(num_samples):
        p = random_point_in_triangle(*random.sample(list(vertices), 3))
        distances.append(np.linalg.norm(p - barycenter))
    return distances

# Function to compute D2 descriptor (distance between 2 random points inside random triangles)
def compute_d2(vertices, num_samples):
    distances = []
    for _ in range(num_samples):
        p1 = random_point_in_triangle(*random.sample(list(vertices), 3))
        p2 = random_point_in_triangle(*random.sample(list(vertices), 3))
        distances.append(np.linalg.norm(p1 - p2))
    return distances

# Function to compute D3 descriptor (square root of area of triangle given by 3 random points)
def compute_d3(vertices, num_samples):
    areas = []
    for _ in range(num_samples):
        p1, p2, p3 = [random_point_in_triangle(*random.sample(list(vertices), 3)) for _ in range(3)]
        a, b, c = np.linalg.norm(p1 - p2), np.linalg.norm(p2 - p3), np.linalg.norm(p3 - p1)
        s = (a + b + c) / 2
        areas.append(np.sqrt(s * (s - a) * (s - b) * (s - c)))
    return areas

# Function to compute D4 descriptor (cube root of volume of tetrahedron formed by 4 random points)
def compute_d4(vertices, num_samples):
    volumes = []
    for _ in range(num_samples):
        p1, p2, p3, p4 = [random_point_in_triangle(*random.sample(list(vertices), 3)) for _ in range(4)]
        volume = np.abs(np.linalg.det(np.vstack((p2 - p1, p3 - p1, p4 - p1)).T)) / 6.0
        volumes.append(np.cbrt(volume))
    return volumes

# Function to compute all descriptors and save them to CSV
def compute_shape_descriptors(file_path, num_samples, output_csv, category, model_name):
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)

    # Compute the raw data for each descriptor
    a3_data = compute_a3(vertices, num_samples)
    d1_data = compute_d1(vertices, num_samples)
    d2_data = compute_d2(vertices, num_samples)
    d3_data = compute_d3(vertices, num_samples)
    d4_data = compute_d4(vertices, num_samples)

    # Write the data to CSV
    with open(output_csv, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for i in range(num_samples):
            writer.writerow([category, model_name, a3_data[i], d1_data[i], d2_data[i], d3_data[i], d4_data[i]])

def process_database(database_path, num_samples=1000, output_csv='shape_descriptors.csv'):
    # Create a new CSV file with a header
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['category', 'model', 'A3', 'D1', 'D2', 'D3', 'D4'])

    for category in os.listdir(database_path):
        category_path = os.path.join(database_path, category)
        if os.path.isdir(category_path):
            for model in os.listdir(category_path):
                if model.endswith('.obj'):
                    model_path = os.path.join(category_path, model)
                    print(f"Processing {model} in {category}...")
                    compute_shape_descriptors(model_path, num_samples, output_csv, category, model)

if __name__ == "__main__":
    # Set the path to your shape database
    database_path = 'ShapeDatabase_INFOMR-master'
    num_samples = 1000  # Modify as needed
    output_csv = 'shape_descriptors.csv'

    # Process the database and save descriptors to CSV
    process_database(database_path, num_samples, output_csv)
    print(f"Processing complete. Results saved to {output_csv}")
