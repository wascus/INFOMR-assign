import random
import numpy as np

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
    if r1 + r2 > 1:
        r1, r2 = 1 - r1, 1 - r2
    return (1 - r1 - r2) * v1 + r1 * v2 + r2 * v3

def compute_d1(vertices, num_samples):
    barycenter = np.mean(vertices, axis=0)
    distances = []
    for _ in range(num_samples):
        v1, v2, v3 = random.sample(list(vertices), 3)
        point = random_point_in_triangle(v1, v2, v3)
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

# ShapeDescriptors function
def ShapeDescriptors(mesh, num_samples=5000, category=None, model_name=None):
    """
    Compute shape descriptors for an Open3D mesh.

    :param mesh: Open3D mesh object to process.
    :param num_samples: Number of samples for each descriptor (default: 5000).
    :param category: Category of the mesh (optional).
    :param model_name: Name of the mesh model (optional).
    :return: List of rows where each row is [category, model, A3, D1, D2, D3, D4].
    """
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)

    # Compute descriptors
    a3_data = compute_a3(vertices, num_samples)
    d1_data = compute_d1(vertices, num_samples)
    d2_data = compute_d2(vertices, num_samples)
    d3_data = compute_d3(vertices, num_samples)
    d4_data = compute_d4(vertices, num_samples)

    # Prepare the raw data
    descriptor_data = [
        [category, model_name, a3_data[i], d1_data[i], d2_data[i], d3_data[i], d4_data[i]]
        for i in range(num_samples)
    ]

    return descriptor_data
