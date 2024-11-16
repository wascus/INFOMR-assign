import tkinter as tk
from tkinter import messagebox, Listbox, filedialog
import os
import pandas as pd
import numpy as np
import open3d as o3d
from sklearn.neighbors import KNeighborsClassifier
from pyemd import emd
from Logic.SearchEMD import search_with_weighted_emd
from Logic.Subsampling import Subsample
from Logic.Supersampling import Supersample
from Logic.CleanManifold import Clean
from Logic.Normalize import Normalize
from Logic.GlobalDescriptors import GlobalDescriptors
from Logic.ShapeDescriptors import ShapeDescriptors
from Logic.HoleFilling import HoleFilling
from Logic.NormalizeDescriptors import normalize_features

# Load feature database
file_path = "feature_vector.csv"
features_df = pd.read_csv(file_path)
master_folder = "ShapeDatabase_INFOMR-master"

vis = None  # Global variable for the Open3D visualizer
current_file_path = None  # Path to the currently loaded model
interactive_mode = False  # Mode for Open3D visualizer
axes_shown = False  # Tracks whether the world axes are displayed
vis_option = "smoothshade"  # Default visualization mode
background_color = [1, 1, 1]  # Default background color (white)

# Define single-value and histogram features
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
feature_columns = single_value_features + [bin for group in histogram_features.values() for bin in group]

# Train k-NN model (for Script 2 functionality)
top_k = 20
data_matrix = features_df[feature_columns].values
knn_model = KNeighborsClassifier(n_neighbors=top_k + 1, metric='euclidean')
knn_model.fit(data_matrix, features_df['Class'])


# Function to retrieve and normalize features from an OBJ file
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

        print("-----------------------------------")
        print(new_model_features)
        return new_model_features

    except Exception as e:
        print(f"Error processing {obj_file_path}: {e}")
        return None


# Function to extract histogram vector
def get_histogram_vector(row):
    histogram_vector = []
    for feature_prefix in ['A3', 'D1', 'D2', 'D3', 'D4']:
        bins = [f'{feature_prefix}_bin_{i}' for i in range(40)]
        histogram_vector.extend(row[bins].values)
    return np.array(histogram_vector)


# Button 1: Search with EMD logic (as per provided script)

def search_emd():
    obj_file_path = filedialog.askopenfilename(filetypes=[("OBJ Files", "*.obj")])
    search_with_weighted_emd(obj_file_path, listbox)


# Button 2: Search with k-NN (unchanged)
def search_knn():
    obj_file_path = filedialog.askopenfilename(filetypes=[("OBJ Files", "*.obj")])
    if not obj_file_path:
        return

    features = modelLineRetrieval(obj_file_path)
    if not features:
        messagebox.showerror("Error", "Failed to process the selected OBJ file.")
        return

    query_vector = np.array([features[col] for col in feature_columns]).reshape(1, -1)
    distances, indices = knn_model.kneighbors(query_vector)

    listbox.delete(0, tk.END)
    for idx, (distance, index) in enumerate(zip(distances[0], indices[0])):
        if idx == 0:
            continue
        matched_file = features_df.iloc[index]['File']
        matched_class = features_df.iloc[index]['Class']
        listbox.insert(tk.END, f"{matched_class} - {matched_file} (Distance: {distance:.4f})")


# GUI Creation
def create_gui():
    global listbox, vis, mode_button, current_file_path

    root = tk.Tk()
    root.title("3D Model Search")

    tk.Label(root, text="3D Model Search Tool").pack(pady=5)

    tk.Button(root, text="Search with EMD/Euc", command=search_emd).pack(pady=10)
    tk.Button(root, text="Search with k-NN", command=search_knn).pack(pady=10)

    listbox = Listbox(root, width=80, height=20)
    listbox.pack(pady=10)
    listbox.bind('<<ListboxSelect>>', on_model_select)

    # Visualization control buttons at the bottom of the window
    control_frame = tk.Frame(root)
    control_frame.pack(pady=10, fill=tk.X)

    tk.Button(control_frame, text="Smooth Shade", command=lambda: set_vis_option("smoothshade")).pack(side=tk.LEFT,
                                                                                                      padx=5)
    tk.Button(control_frame, text="Wireframe", command=lambda: set_vis_option("wireframe")).pack(side=tk.LEFT, padx=5)
    tk.Button(control_frame, text="Wireframe on Shaded", command=lambda: set_vis_option("wireframe_on_shaded")).pack(
        side=tk.LEFT, padx=5)
    tk.Button(control_frame, text="World Axes", command=lambda: set_vis_option("world_axes")).pack(side=tk.LEFT, padx=5)
    tk.Button(control_frame, text="Toggle Background", command=toggle_background).pack(side=tk.LEFT, padx=5)

    additional_control_frame = tk.Frame(root)
    additional_control_frame.pack(pady=10, fill=tk.X)

    mode_button = tk.Button(additional_control_frame, text="Switch to Interactive", command=toggle_mode)
    mode_button.pack(side=tk.LEFT, padx=5)
    tk.Button(additional_control_frame, text="Turn Off Viewer", command=turn_off_visualizer).pack(side=tk.LEFT, padx=5)
    tk.Button(additional_control_frame, text="Reset", command=reset_viewer).pack(side=tk.LEFT, padx=5)

    root.mainloop()


def on_model_select(event):
    global current_file_path

    selected_idx = listbox.curselection()
    if selected_idx:
        selected_item = listbox.get(selected_idx[0])

        # Extract class and model file name from the listbox item
        try:
            class_name = selected_item.split(' ')[0]
            model_file = selected_item.split(' - ')[1].split(' (')[0]

            # Construct the full path to the model
            model_path = os.path.join(master_folder, class_name, model_file)

            # Check if the file exists before attempting to load
            if os.path.exists(model_path):
                current_file_path = model_path
                load_and_view_model(model_path)
            else:
                messagebox.showerror("Error", f"Model file '{model_path}' not found.")
        except IndexError:
            messagebox.showerror("Error", "Could not parse the selected item. Please check the list format.")


def set_vis_option(option):
    global vis_option, axes_shown
    vis_option = option

    if current_file_path:
        if option == "world_axes":
            axes_shown = not axes_shown  # Toggle axes visibility
        load_and_view_model(current_file_path)


def load_and_view_model(file_path):
    global vis, interactive_mode, vis_option, background_color, axes_shown

    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Model Viewer", width=800, height=600)

    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    mesh = normalize_mesh(mesh)

    # Clear geometries
    vis.clear_geometries()

    # Handle visual options
    if vis_option == "smoothshade":
        vis.add_geometry(mesh)
        vis.get_render_option().mesh_show_wireframe = False
    elif vis_option == "wireframe":
        vis.add_geometry(mesh)
        vis.get_render_option().mesh_show_wireframe = True
        vis.get_render_option().mesh_show_back_face = False
        mesh.paint_uniform_color([0, 1, 1])  # Set wireframe color (optional)
    elif vis_option == "wireframe_on_shaded":
        vis.add_geometry(mesh)
        vis.get_render_option().mesh_show_wireframe = True
        vis.get_render_option().mesh_show_back_face = True
    elif vis_option == "world_axes":
        if axes_shown:
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            vis.add_geometry(axes)

    # Set render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray(background_color)

    # Render the model
    if interactive_mode:
        vis.run()
        vis.destroy_window()
        vis = None
    else:
        vis.poll_events()
        vis.update_renderer()


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


def turn_off_visualizer():
    global vis
    if vis:
        vis.destroy_window()
        vis = None


def reset_viewer():
    global current_file_path
    listbox.delete(0, tk.END)
    current_file_path = None
    turn_off_visualizer()


if __name__ == "__main__":
    create_gui()
