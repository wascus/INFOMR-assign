import pandas as pd
import numpy as np
from pyemd import emd
import tkinter as tk
from tkinter import messagebox, Listbox
import open3d as o3d
import os

# Load the normalized features from the provided CSV file
file_path = "merged_normalized_features_combined.csv"
master_folder = "ShapeDatabase_INFOMR-master"  # Replace with your master folder path
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

# Global variables for Open3D visualizer and settings
vis = None
current_file_path = None
interactive_mode = False
vis_option = "smoothshade"
background_color = [1, 1, 1]  # Default to white background
show_axes = False  # Toggle for displaying world axes
axes_geometry = None  # Store the axes geometry to toggle it

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

def search_similar_models():
    query_shape_filename = entry.get()
    query_row = features_df[features_df['File'] == query_shape_filename]

    if query_row.empty:
        messagebox.showerror("Error", f"Shape '{query_shape_filename}' not found in the dataset.")
    else:
        query_vector = query_row[feature_columns].values.flatten()
        features_matrix = features_df[feature_columns].values

        # Calculate distance ranges for each feature in the dataset
        distance_ranges = calculate_distance_ranges(features_matrix)

        # Calculate EMD distances with distance weighting
        emd_distances = calculate_emd(query_vector, features_matrix, distance_ranges)
        features_df['Distance'] = emd_distances

        # Sort by distance and get the top 20 matches
        top_matches = features_df.sort_values(by='Distance').head(20)

        # Update the listbox with the results
        listbox.delete(0, tk.END)
        for idx, row in top_matches.iterrows():
            listbox.insert(tk.END, f"{row['Class']} - {row['File']} (Distance: {row['Distance']:.4f})")

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
    global show_axes
    show_axes = not show_axes
    if current_file_path:
        load_and_view_model(current_file_path)

def turn_off_visualizer():
    global vis
    if vis:
        vis.destroy_window()
        vis = None
        print("Visualizer turned off.")

def reset_viewer():
    global current_file_path, axes_geometry
    listbox.delete(0, tk.END)
    current_file_path = None
    turn_off_visualizer()
    axes_geometry = None
    print("Viewer and list reset.")

def create_gui():
    global entry, listbox, mode_button

    root = tk.Tk()
    root.title("3D Model Search Using EMD with Open3D Viewer")

    # Add input field and search button
    tk.Label(root, text="Enter Model Name:").pack(pady=5)
    entry = tk.Entry(root)
    entry.pack(pady=5)

    search_button = tk.Button(root, text="Search", command=search_similar_models)
    search_button.pack(pady=5)

    # Add a listbox to display the top 20 results
    listbox = Listbox(root, width=80, height=20)
    listbox.pack(pady=10)
    listbox.bind('<<ListboxSelect>>', on_model_select)

    # Add buttons for visualization options
    tk.Button(root, text="Smooth Shade", command=lambda: set_vis_option("smoothshade")).pack(fill=tk.X)
    tk.Button(root, text="Wireframe", command=lambda: set_vis_option("wireframe")).pack(fill=tk.X)
    tk.Button(root, text="Wireframe on Shaded", command=lambda: set_vis_option("wireframe_on_shaded")).pack(fill=tk.X)
    tk.Button(root, text="Toggle World Axes", command=toggle_axes).pack(fill=tk.X)
    tk.Button(root, text="Toggle Background", command=toggle_background).pack(fill=tk.X)

    # Add buttons for toggling mode, turning off the viewer, and resetting
    mode_button = tk.Button(root, text="Switch to Interactive", command=toggle_mode)
    mode_button.pack(pady=5)
    tk.Button(root, text="Turn Off Viewer", command=turn_off_visualizer).pack(pady=5)
    tk.Button(root, text="Reset", command=reset_viewer).pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    create_gui()