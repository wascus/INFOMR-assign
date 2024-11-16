import tkinter as tk
from tkinter import messagebox, Listbox, filedialog
import pandas as pd
import numpy as np
import open3d as o3d
from pyemd import emd
import os
import pickle
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance

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
histogram_features = {
    'A3': [f'A3_bin_{i}' for i in range(40)],
    'D1': [f'D1_bin_{i}' for i in range(40)],
    'D2': [f'D2_bin_{i}' for i in range(40)],
    'D3': [f'D3_bin_{i}' for i in range(40)],
    'D4': [f'D4_bin_{i}' for i in range(40)]
}

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

feature_weights = {
    'A3': 1.0,
    'D1': 1000.0,
    'D2': 1000.0,
    'D3': 1000.0,
    'D4': 1000.0
}

def get_histogram_vector(row):
    histogram_vector = []
    for feature_prefix in ['A3', 'D1', 'D2', 'D3', 'D4']:
        bins = [f'{feature_prefix}_bin_{i}' for i in range(40)]
        histogram_vector.extend(row[bins].values)
    return np.array(histogram_vector)

# Define the distance function using the weighted EMD and Euclidean distance
def compute_distance(query, candidate, histogram_features, single_value_features, feature_weights):
    # EMD for histogram features
    weighted_emd_sum = 0
    for feature, bins in histogram_features.items():
        query_hist = query[bins].values
        candidate_hist = candidate[bins].values
        emd = wasserstein_distance(query_hist, candidate_hist)
        weighted_emd_sum += feature_weights[feature] * emd

    # Euclidean distance for single-value features
    query_values = query[single_value_features].values
    candidate_values = candidate[single_value_features].values
    euclidean_dist = np.sqrt(np.sum((query_values - candidate_values) ** 2))

    return weighted_emd_sum * euclidean_dist

# Find the closest matches using the new distance function
def query_shape(query_id, features_df, histogram_features, single_value_features, feature_weights, top_k=10):
    query = features_df.loc[features_df['File'] == query_id].iloc[0]
    distances = []
    for index, row in features_df.iterrows():
        if row['File'] != query_id:
            dist = compute_distance(query, row, histogram_features, single_value_features, feature_weights)
            distances.append((row['File'], row['Class'], dist))
    distances.sort(key=lambda x: x[2])
    return distances[:top_k]

# Adjusted search function to display results in the GUI
def search_with_weighted_emd(path, listbox):
    obj_file_path = path
    if not obj_file_path:
        return

    new_model_features = modelLineRetrieval(obj_file_path)
    if not new_model_features:
        messagebox.showerror("Error", "Could not process the selected OBJ file.")
        return

    # Append new model features temporarily to the DataFrame for querying
    query_row = pd.DataFrame([new_model_features])
    temp_features_df = pd.concat([features_df, query_row], ignore_index=True)

    # Perform the search with weighted EMD and Euclidean distance
    top_matches = query_shape("0", temp_features_df, histogram_features, single_value_features, feature_weights, top_k=20)

    # Update the listbox with the top matches
    listbox.delete(0, tk.END)
    for file, class_label, dist in top_matches:
        listbox.insert(tk.END, f"{class_label} - {file} (Distance: {dist:.4f})")
