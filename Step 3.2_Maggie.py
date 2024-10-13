import trimesh
import numpy as np
import random
import os
import csv

def load_mesh(file_path):
    #use trimesh to load the mesh
    mesh = trimesh.load(file_path)
    vertices = np.array(mesh.vertices)
    return mesh, vertices


def compute_surface_area(mesh):
    return mesh.area


def compute_compactness(mesh, surface_area):
    volume = mesh.volume  #approximation for non-watertight meshes
    if volume == 0:
        return 0
    compactness = (surface_area ** 3) / (36 * np.pi * volume ** 2)
    return compactness


def compute_rectangularity(mesh, volume):
    obb = mesh.bounding_box_oriented
    obb_volume = obb.volume
    if obb_volume == 0:
        return 0
    rectangularity = volume / obb_volume
    return rectangularity


def compute_diameter(vertices):
    pairwise_distances = np.linalg.norm(vertices[:, np.newaxis] - vertices, axis=-1)
    return np.max(pairwise_distances)


def compute_convexity(mesh):
    convex_hull = mesh.convex_hull
    convex_hull_volume = convex_hull.volume
    mesh_volume = mesh.volume
    if convex_hull_volume == 0:
        return 0
    convexity = mesh_volume / convex_hull_volume
    return convexity


def compute_eccentricity(vertices):
    covariance_matrix = np.cov(vertices.T)
    eigenvalues, _ = np.linalg.eig(covariance_matrix)
    if min(eigenvalues) == 0:
        return 0
    eccentricity = max(eigenvalues) / min(eigenvalues)
    return eccentricity


def compute_global_property_descriptors(file_path):
    mesh, vertices = load_mesh(file_path)

    #compute all global shape descriptors
    surface_area = compute_surface_area(mesh)
    volume = mesh.volume  # Trimesh handles non-watertight volumes
    compactness = compute_compactness(mesh, surface_area)
    rectangularity = compute_rectangularity(mesh, volume)
    diameter = compute_diameter(vertices)
    convexity = compute_convexity(mesh)
    eccentricity = compute_eccentricity(vertices)

    #concatenate into a single descriptor vector
    global_property_descriptors = np.concatenate([
        [surface_area, compactness, rectangularity, diameter, convexity, eccentricity]
    ])

    return global_property_descriptors

#iterate over all categories and compute descriptors for each file
def process_database(database_path, output_csv='global_property_descriptors.csv'):
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        header = ['category', 'model', 'surface_area', 'compactness', 'rectangularity', 'diameter', 'convexity', 'eccentricity']
        writer.writerow(header)

        for category in os.listdir(database_path):
            category_path = os.path.join(database_path, category)
            if os.path.isdir(category_path):  #find folders representing categories
                for model in os.listdir(category_path):
                    if model.endswith('.obj'):  #find the .obj mesh files
                        model_path = os.path.join(category_path, model)
                        descriptor = compute_global_property_descriptors(model_path)
                        writer.writerow([category, model] + list(descriptor))


if __name__ == "__main__":

    database_path = '/Users/maggiemaliszewski/INFOMR-assign/ShapeDatabase_INFOMR-master' #update accordingly

    #process the entire database and save results to a csv file
    output_csv = 'global_property_descriptors.csv'
    process_database(database_path, output_csv)
