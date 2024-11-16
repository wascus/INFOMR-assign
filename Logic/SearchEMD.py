import tkinter as tk
from tkinter import messagebox, Listbox, filedialog
import pandas as pd
import numpy as np
import open3d as o3d
from pyemd import emd
import os
import pickle
from scipy.spatial.distance import pdist

from Logic.Subsampling import Subsample
from Logic.Supersampling import Supersample
from Logic.CleanManifold import Clean
from Logic.Normalize import Normalize
from Logic.GlobalDescriptors import GlobalDescriptors
from Logic.ShapeDescriptors import ShapeDescriptors
from Logic.HoleFilling import HoleFilling
from Logic.NormalizeDescriptors import normalize_features

file_path = "feature_vector.csv"
features_df = pd.read_csv(file_path)
single_value_features = [
    "Surface Area", "Volume", "Compactness", "Rectangularity", "Diameter",
    "Convexity", "Eccentricity"
]

def modelLineRetrieval(obj_file_path):
    try:
        # Load and process the OBJ file
        mesh = o3d.io.read_triangle_mesh(obj_file_path)
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        num_faces = len(mesh.triangles)
        if num_faces > 13000:
            mesh = Subsample(mesh)
        elif num_faces < 9000:
            mesh = Supersample(mesh)

        mesh = Clean(mesh)
        mesh = Normalize(mesh)
        mesh = HoleFilling(mesh)

        global_descriptors = GlobalDescriptors(mesh)
        shape_descriptors = ShapeDescriptors(mesh)

        a3_values = [row[2] for row in shape_descriptors]
        d1_values = [row[3] for row in shape_descriptors]
        d2_values = [row[4] for row in shape_descriptors]
        d3_values = [row[5] for row in shape_descriptors]
        d4_values = [row[6] for row in shape_descriptors]

        features = normalize_features("0", global_descriptors, a3_values, d1_values, d2_values, d3_values, d4_values)

        # Expand histogram features into individual columns
        histogram_features = {}
        for i, value in enumerate(features["A3"]):
            histogram_features[f"A3_bin_{i}"] = value
        for i, value in enumerate(features["D1"]):
            histogram_features[f"D1_bin_{i}"] = value
        for i, value in enumerate(features["D2"]):
            histogram_features[f"D2_bin_{i}"] = value
        for i, value in enumerate(features["D3"]):
            histogram_features[f"D3_bin_{i}"] = value
        for i, value in enumerate(features["D4"]):
            histogram_features[f"D4_bin_{i}"] = value

        # Merge global and histogram features
        new_model_features = {
            "File": "0",
            "Surface Area": features["Surface Area"],
            "Volume": features["Volume"],
            "Compactness": features["Compactness"],
            "Rectangularity": features["Rectangularity"],
            "Diameter": features["Diameter"],
            "Convexity": features["Convexity"],
            "Eccentricity": features["Eccentricity"],
            **histogram_features
        }

        return new_model_features

    except Exception as e:
        print(f"Error processing {obj_file_path}: {e}")
        return None

def get_histogram_vector(row):
    histogram_vector = []
    for feature_prefix in ['A3', 'D1', 'D2', 'D3', 'D4']:
        bins = [f'{feature_prefix}_bin_{i}' for i in range(40)]
        histogram_vector.extend(row[bins].values)
    return np.array(histogram_vector)

def calculate_l2_distance(query_vector, features_matrix):
    return np.sqrt(np.sum((features_matrix - query_vector) ** 2, axis=1))

# Calculate EMD for histogram-based features
def calculate_emd(query_vector, features_matrix, distance_ranges):
    emd_distances = []
    normalized_query = query_vector / distance_ranges
    for feature_vector in features_matrix:
        normalized_feature_vector = feature_vector / distance_ranges
        distance_matrix = np.ones((len(query_vector), len(query_vector))) - np.eye(len(query_vector))
        emd_distances.append(
            emd(normalized_query.astype(np.float64), normalized_feature_vector.astype(np.float64), distance_matrix)
        )
    return emd_distances

# Precompute distance ranges for histogram features
def calculate_distance_ranges(features_matrix):
    ranges = []
    for i in range(features_matrix.shape[1]):
        feature_column = features_matrix[:, i]
        distances = np.abs(feature_column[:, None] - feature_column)
        range_val = np.max(distances) - np.min(distances)
        ranges.append(range_val if range_val != 0 else 1)
    return np.array(ranges)

def search_with_emd(path, listbox):
    obj_file_path = path
    if not obj_file_path:
        return

    new_model_features = modelLineRetrieval(obj_file_path)
    if not new_model_features:
        messagebox.showerror("Error", "Could not process the selected OBJ file.")
        return

    query_row = pd.DataFrame([new_model_features])
    query_vector_single = query_row[single_value_features].values.flatten()
    query_vector_histogram = get_histogram_vector(query_row.iloc[0])

    features_matrix_single = features_df[single_value_features].values
    features_matrix_histogram = np.array([get_histogram_vector(row) for _, row in features_df.iterrows()])

    l2_distances = calculate_l2_distance(query_vector_single, features_matrix_single)
    distance_ranges = calculate_distance_ranges(features_matrix_histogram)
    emd_distances = calculate_emd(query_vector_histogram, features_matrix_histogram, distance_ranges)

    features_df['Distance'] = l2_distances * emd_distances
    top_matches = features_df.sort_values(by='Distance').iloc[:20]

    listbox.delete(0, tk.END)
    for _, row in top_matches.iterrows():
        listbox.insert(tk.END, f"{row['Class']} - {row['File']} (Distance: {row['Distance']:.4f})")

# Extend distance ranges for global descriptors
def calculate_distance_ranges_with_globals(features_df, single_value_features, features_matrix_histogram):
    """Calculate distance ranges for both single-value and histogram features."""
    # For single-value features, calculate min-max ranges
    ranges_global = features_df[single_value_features].max() - features_df[single_value_features].min()
    ranges_global = ranges_global.replace(0, 1)  # Avoid division by zero

    # For histogram features, calculate histogram distance ranges
    ranges_histogram = calculate_distance_ranges(features_matrix_histogram)

    return ranges_global, ranges_histogram

# Normalize both query and database features using ranges
def search_with_emd_using_ranges(path, listbox):
    global features_df  # Ensure you modify the global variable features_df

    obj_file_path = path
    if not obj_file_path:
        return

    new_model_features = modelLineRetrieval(obj_file_path)
    if not new_model_features:
        messagebox.showerror("Error", "Could not process the selected OBJ file.")
        return

    query_row = pd.DataFrame([new_model_features])

    # Calculate distance ranges for normalization
    features_matrix_histogram = np.array([get_histogram_vector(row) for _, row in features_df.iterrows()])
    ranges_global, ranges_histogram = calculate_distance_ranges_with_globals(features_df, single_value_features, features_matrix_histogram)

    # Normalize database and query global features using ranges
    features_df[single_value_features] = features_df[single_value_features] / ranges_global
    query_row[single_value_features] = query_row[single_value_features] / ranges_global

    # Normalize histogram features
    query_vector_histogram = get_histogram_vector(query_row.iloc[0]) / ranges_histogram
    features_matrix_histogram = features_matrix_histogram / ranges_histogram

    # Calculate distances
    query_vector_single = query_row[single_value_features].values.flatten()
    features_matrix_single = features_df[single_value_features].values

    l2_distances = calculate_l2_distance(query_vector_single, features_matrix_single)
    emd_distances = calculate_emd(query_vector_histogram, features_matrix_histogram, np.ones_like(ranges_histogram))

    features_df['Distance'] = l2_distances * emd_distances
    top_matches = features_df.sort_values(by='Distance').iloc[:20]

    listbox.delete(0, tk.END)
    for _, row in top_matches.iterrows():
        listbox.insert(tk.END, f"{row['Class']} - {row['File']} (Distance: {row['Distance']:.4f})")

