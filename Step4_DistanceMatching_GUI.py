import pandas as pd
import numpy as np
from pyemd import emd
import tkinter as tk
from tkinter import messagebox, Listbox
import open3d as o3d
import os
from scipy.spatial.distance import cosine

# Load the normalized features from the provided CSV file
file_path = "feature_vector.csv"
master_folder = "ShapeDatabase_INFOMR-master"  # Replace with your master folder path
features_df = pd.read_csv(file_path)

# Define the columns representing the features
single_value_features = ["Surface Area", "Volume", "Compactness", "Rectangularity", "Diameter",
                         "Convexity", "Eccentricity"]
histogram_groups = {
    'A3': [f'A3_bin_{i}.0' for i in range(10)],
    'D1': [f'D1_bin_{i}.0' for i in range(10)],
    'D2': [f'D2_bin_{i}.0' for i in range(10)],
    'D3': [f'D3_bin_{i}.0' for i in range(10)],
    'D4': [f'D4_bin_{i}.0' for i in range(10)]
}

# Precompute distance ranges per feature for distance normalization
def calculate_distance_ranges(features_matrix):
    ranges = []
    for i in range(features_matrix.shape[1]):
        feature_column = features_matrix[:, i]
        distances = np.abs(feature_column[:, None] - feature_column)  # Pairwise distances
        range_val = np.max(distances) - np.min(distances)  # Range of distances for this feature
        ranges.append(range_val if range_val != 0 else 1)  # Avoid division by zero
    return np.array(ranges)

# Helper function to compute L2 distance for single-value features
def calculate_l2(row_a, row_b, features):
    return np.linalg.norm(row_a[features].values - row_b[features].values)

# Helper function to compute EMD for histogram groups
def calculate_emd(query_vector, features_matrix, distance_ranges, feature_bins):
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

# Combined function for calculating weighted distance
def combined_distance(row_a, row_b, single_value_features, histogram_groups):
    total_distance = 0.0

    # L2 distance for single-value features
    l2_dist = calculate_l2(row_a, row_b, single_value_features)
    total_distance += l2_dist

    # EMD for histogram groups
    for group_name, feature_bins in histogram_groups.items():
        emd_dist = calculate_emd(row_a[feature_bins], row_b[feature_bins], feature_bins)
        total_distance += emd_dist

    return total_distance

# Global variables for Open3D visualizer and settings
vis = None
current_file_path = None
interactive_mode = False
vis_option = "smoothshade"
background_color = [1, 1, 1]  # Default to white background
show_axes = False  # Toggle for displaying world axes
axes_geometry = None  # Store the axes geometry to toggle it

# Calculate EMD with distance weighting
def search_similar_models():
    query_shape_filename = entry.get()
    query_row = features_df[features_df['File'] == query_shape_filename]

    if query_row.empty:
        messagebox.showerror("Error", f"Shape '{query_shape_filename}' not found in the dataset.")
    else:
        query_vector = query_row[single_value_features].values.flatten()
        features_matrix = features_df[single_value_features].values

        # Calculate distance ranges for each feature in the dataset
        distance_ranges = calculate_distance_ranges(features_matrix)

        # Calculate combined distances with distance weighting
        distances = []
        for i, row in features_df.iterrows():
            dist = combined_distance(query_row.iloc[0], row, single_value_features, histogram_groups)
            distances.append((row['File'], row['Class'], dist))

        # Sort by distance and get the top 20 matches
        top_matches = sorted(distances, key=lambda x: x[2])[:20]

        # Update the listbox with the results
        listbox.delete(0, tk.END)
        for match in top_matches:
            listbox.insert(tk.END, f"{match[1]} - {match[0]} (Distance: {match[2]:.4f})")

def on_model_select(event):
    global current_file_path

    selected_idx = listbox.curselection()
    if selected_idx:
        selected_item = listbox.get(selected_idx[0])
        model_file = selected_item.split(' - ')[1].split(' (')[0]
        category = selected_item.split(' ')[0]
        model_path = os.path.join(master_folder, category, model_file)

        if os.path.exists(model_path):
            current_file_path = model_path
            load_and_view_model(model_path)
        else:
            messagebox.showerror("Error", f"Model file '{model_path}' not found.")

def load_and_view_model(file_path):
    global vis, interactive_mode, vis_option, background_color, show_axes, axes_geometry
    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Model Viewer", width=800, height=600)

    # Read the mesh
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Normalize the mesh (scale and center)
    mesh = normalize_mesh(mesh)

    # Clear previous geometries
    vis.clear_geometries()

    # Add geometry based on visualization option
    if vis_option == "smoothshade":
        vis.add_geometry(mesh)
        vis.get_render_option().mesh_show_wireframe = False
    elif vis_option == "wireframe":
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        vis.add_geometry(wireframe)
        vis.get_render_option().mesh_show_wireframe = False
    elif vis_option == "wireframe_on_shaded":
        vis.add_geometry(mesh)
        vis.get_render_option().mesh_show_wireframe = True

    # Toggle the world axes visibility
    if show_axes:
        if axes_geometry is None:
            axes_geometry = create_thin_axes()
        vis.add_geometry(axes_geometry)

    # Set the background color
    opt = vis.get_render_option()
    opt.background_color = np.asarray(background_color)

    # Update the visualizer based on the current mode
    if interactive_mode:
        vis.run()
        vis.destroy_window()
        vis = None
    else:
        vis.poll_events()
        vis.update_renderer()

def create_thin_axes(size=0.5, thickness=0.01):
    points = [
        [0, 0, 0], [size, 0, 0],  # X-axis
        [0, 0, 0], [0, size, 0],  # Y-axis
        [0, 0, 0], [0, 0, size]   # Z-axis
    ]
    lines = [[0, 1], [2, 3], [4, 5]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB colors for X, Y, Z
    axes = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    axes.colors = o3d.utility.Vector3dVector(colors)
    return axes

def normalize_mesh(mesh):
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    mesh.translate(-center)
    extent = bbox.get_extent()
    scale_factor = 1.0 / max(extent)
    mesh.scale(scale_factor, center=[0, 0, 0])
    return mesh

def toggle_mode():
    global interactive_mode
    interactive_mode = not interactive_mode
    mode_button.config(text="Switch to Interactive" if not interactive_mode else "Switch to Automatic")
    print(f"Mode switched to {'Interactive' if interactive_mode else 'Automatic'}")

def set_vis_option(option):
    global vis_option
    vis_option = option
    if current_file_path:
        load_and_view_model(current_file_path)

def toggle_background():
    global background_color
    background_color = [0, 0, 0] if background_color == [1, 1, 1] else [1, 1, 1]
    if current_file_path:
        load_and_view_model(current_file_path)

def toggle_axes():
    global show_axes, axes_geometry
    show_axes = not show_axes
    if axes_geometry:
        vis.remove_geometry(axes_geometry)
        axes_geometry = None
    if show_axes:
        axes_geometry = create_thin_axes()
        vis.add_geometry(axes_geometry)
    if current_file_path:
        load_and_view_model(current_file_path)

def reset_view():
    if vis:
        vis.reset_view_point()

# Create the GUI
root = tk.Tk()
root.title("3D Shape Retrieval System")

# Input for querying a shape file
label = tk.Label(root, text="Enter shape file name:")
label.pack(padx=10, pady=5)

entry = tk.Entry(root, width=50)
entry.pack(padx=10, pady=5)

search_button = tk.Button(root, text="Search", command=search_similar_models)
search_button.pack(padx=10, pady=5)

# Listbox to show search results
listbox = Listbox(root, width=50, height=20)
listbox.pack(padx=10, pady=5)
listbox.bind("<Double-1>", on_model_select)

# Buttons for visualization options
mode_button = tk.Button(root, text="Switch to Interactive", command=toggle_mode)
mode_button.pack(padx=10, pady=5)

background_button = tk.Button(root, text="Toggle Background", command=toggle_background)
background_button.pack(padx=10, pady=5)

axes_button = tk.Button(root, text="Toggle Axes", command=toggle_axes)
axes_button.pack(padx=10, pady=5)

reset_button = tk.Button(root, text="Reset View", command=reset_view)
reset_button.pack(padx=10, pady=5)

# Run the GUI
root.mainloop()
