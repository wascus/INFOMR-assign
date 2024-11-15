import pandas as pd
import numpy as np
from pyemd import emd
import tkinter as tk
from tkinter import messagebox, Listbox, filedialog
import open3d as o3d
import os
from Logic.Subsampling import Subsample
from Logic.Supersampling import Supersample
from Logic.CleanManifold import Clean
from Logic.Normalize import Normalize

# Load the normalized features from the provided CSV file
file_path = "merged_normalized_features_combined.csv"
features_df = pd.read_csv(file_path)

# Define the columns representing the features
feature_columns = [
    "Surface Area", "Volume", "Compactness", "Rectangularity", "Diameter",
    "Convexity", "Eccentricity", "A3", "D1", "D2", "D3", "D4"
]


# Precompute distance ranges per feature for distance normalization
def calculate_distance_ranges(features_matrix):
    ranges = []
    for i in range(features_matrix.shape[1]):
        feature_column = features_matrix[:, i]
        distances = np.abs(feature_column[:, None] - feature_column)  # Pairwise distances
        range_val = np.max(distances) - np.min(distances)  # Range of distances for this feature
        ranges.append(range_val if range_val != 0 else 1)  # Avoid division by zero
    return np.array(ranges)


# Feature extraction method for the model line retrieval
def modelLineRetrieval(obj_file_path):
    try:
        # Load the OBJ file into an Open3D mesh
        mesh = o3d.io.read_triangle_mesh(obj_file_path)
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        # Step 2: Resample
        num_faces = len(mesh.triangles)
        if num_faces > 13000:
            mesh = Subsample(mesh)  # Placeholder for decimate function
        elif num_faces < 9000:
            mesh = Supersample(mesh)  # Placeholder for subdivide function
        else:
            mesh = Clean(mesh)  # Placeholder for clean function

        # Step 3: Normalize
        mesh = Normalize(mesh)

        #Step 4: HoleFilling

        #Step 5: Get

        # Extract various features (implementations will be handled by another script)
        surface_area = calculate_surface_area(mesh)  # Placeholder function
        volume = calculate_volume(mesh)  # Placeholder function
        compactness = calculate_compactness(surface_area, volume)  # Placeholder function
        rectangularity = calculate_rectangularity(mesh)  # Placeholder function
        diameter = calculate_diameter(mesh)  # Placeholder function
        convexity = calculate_convexity(mesh)  # Placeholder function
        eccentricity = calculate_eccentricity(mesh)  # Placeholder function
        a3 = calculate_a3_descriptor(mesh)  # Placeholder function
        d1 = calculate_d1_descriptor(mesh)  # Placeholder function
        d2 = calculate_d2_descriptor(mesh)  # Placeholder function
        d3 = calculate_d3_descriptor(mesh)  # Placeholder function
        d4 = calculate_d4_descriptor(mesh)  # Placeholder function

        # Create a dictionary with all features (matching the format of the CSV)
        new_model_features = {
            "File": "D00514.obj",
            "Surface Area": -0.056155981531512834,
            "Volume": -0.32927574454562075,
            "Compactness": -0.3042277698177236,
            "Rectangularity": -0.5003597158421492,
            "Diameter": 0.6695249597165905,
            "Convexity": -0.5530466625113731,
            "Eccentricity": 0.0009485357542467816,
            "A3": 0.9676700183196667,
            "D1": 0.4614996880820569,
            "D2": 0.4585949265216225,
            "D3": 0.14833945121559433,
            "D4": 0.2661606466514979
        }

        return new_model_features

    except Exception as e:
        print(f"Error processing {obj_file_path}: {e}")
        return None


# Calculate EMD with distance weighting
def calculate_emd(query_vector, features_matrix, distance_ranges):
    distance_matrix = np.ones((len(query_vector), len(query_vector))) - np.eye(len(query_vector))
    emd_distances = []
    for feature_vector in features_matrix:
        # Apply distance weighting by dividing each feature by its precomputed range
        normalized_query = query_vector / distance_ranges
        normalized_feature_vector = feature_vector / distance_ranges
        emd_distances.append(
            emd(normalized_query.astype(np.float64), normalized_feature_vector.astype(np.float64), distance_matrix)
        )
    return emd_distances


# Search for similar models
def search_similar_models():
    # Check if an OBJ file has been selected
    if not current_file_path or not current_file_path.endswith(".obj"):
        messagebox.showerror("Error", "Please select a valid OBJ file.")
        return

    # Run modelLineRetrieval to get the extracted features from the input OBJ file
    new_model_features = modelLineRetrieval(current_file_path)

    if not new_model_features:
        messagebox.showerror("Error", "Could not extract features from the provided OBJ file.")
        return

    # Convert the dictionary of features to a DataFrame for comparison
    query_row = pd.DataFrame([new_model_features])

    # Extract the feature values for the query
    query_vector = query_row[feature_columns].values.flatten()
    features_matrix = features_df[feature_columns].values

    # Calculate distance ranges for each feature in the dataset
    distance_ranges = calculate_distance_ranges(features_matrix)

    # Calculate EMD distances with distance weighting
    emd_distances = calculate_emd(query_vector, features_matrix, distance_ranges)
    features_df['Distance'] = emd_distances

    # Sort by distance and get the top 20 matches
    top_matches = features_df.sort_values(by='Distance').iloc[:20]  # Get the top 20 closest matches

    # Update the listbox with the results
    listbox.delete(0, tk.END)
    for idx, row in top_matches.iterrows():
        listbox.insert(tk.END, f"{row['Class']} - {row['File']} (Distance: {row['Distance']:.4f})")


# Open file dialog to select OBJ file
def browse_file():
    global current_file_path
    # Open a file dialog to select an OBJ file
    file_path = filedialog.askopenfilename(filetypes=[("OBJ Files", "*.obj")])
    if file_path:
        current_file_path = file_path
        entry.config(state="normal")
        entry.delete(0, tk.END)
        entry.insert(0, os.path.basename(file_path))
        entry.config(state="readonly")


# GUI Creation
def create_gui():
    global entry, listbox, current_file_path

    root = tk.Tk()
    root.title("3D Model Search Using EMD with Open3D Viewer")

    # Add input field, browse button, and search button
    tk.Label(root, text="Select OBJ File:").pack(pady=5)
    entry = tk.Entry(root, state="readonly")
    entry.pack(pady=5)

    browse_button = tk.Button(root, text="Browse", command=browse_file)
    browse_button.pack(pady=5)

    search_button = tk.Button(root, text="Search", command=search_similar_models)
    search_button.pack(pady=5)

    # Add a listbox to display the top 20 results
    listbox = Listbox(root, width=80, height=20)
    listbox.pack(pady=10)

    root.mainloop()


# Placeholder utility functions for feature calculations
# Implementations will be handled by another script, and these are just placeholders for now
def calculate_surface_area(mesh):
    pass


def calculate_volume(mesh):
    pass


def calculate_compactness(surface_area, volume):
    pass


def calculate_rectangularity(mesh):
    pass


def calculate_diameter(mesh):
    pass


def calculate_convexity(mesh):
    pass


def calculate_eccentricity(mesh):
    pass


def calculate_a3_descriptor(mesh):
    pass


def calculate_d1_descriptor(mesh):
    pass


def calculate_d2_descriptor(mesh):
    pass


def calculate_d3_descriptor(mesh):
    pass


def calculate_d4_descriptor(mesh):
    pass


if __name__ == "__main__":
    create_gui()
