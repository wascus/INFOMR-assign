import tkinter as tk
from tkinter import ttk
import open3d as o3d
import os
import numpy as np

# Global variables
vis = None
vis_option = "smoothshade"
current_file_path = None
interactive_mode = False  # This variable will control the mode (Automatic or Interactive)


def load_folder_structure(tree, folder_path):

    node = tree.insert("", "end", text=os.path.basename(folder_path), open=True)
    populate_tree(tree, node, folder_path)


def populate_tree(tree, parent, folder):

    for item in os.listdir(folder):
        full_path = os.path.join(folder, item)
        if os.path.isdir(full_path):
            # Insert folder into the treeview
            node = tree.insert(parent, "end", text=item, open=False)
            populate_tree(tree, node, full_path)
        else:
            # Insert file into the treeview
            if item.lower().endswith(('.ply', '.stl', '.obj', '.off')):  # Add other formats as needed
                tree.insert(parent, "end", text=item, values=[full_path])


def create_buttons(right_frame):

    tk.Button(right_frame, text="Smooth Shade", command=lambda: set_vis_option("smoothshade")).pack(fill=tk.X)
    tk.Button(right_frame, text="Wireframe", command=lambda: set_vis_option("wireframe")).pack(fill=tk.X)
    tk.Button(right_frame, text="Wireframe on Shaded", command=lambda: set_vis_option("wireframe_on_shaded")).pack(
        fill=tk.X)
    tk.Button(right_frame, text="World Axes", command=lambda: set_vis_option("world_axes")).pack(fill=tk.X)
    tk.Button(right_frame, text="Black Background", command=lambda: set_vis_option("black_background")).pack(fill=tk.X)

    # Add a button to toggle between Automatic Update and Interactive View
    tk.Button(right_frame, text="Toggle Mode", command=toggle_mode).pack(fill=tk.X)


def set_vis_option(option):

    global vis_option, current_file_path
    vis_option = option
    # If a model is already loaded, re-render it with the new visualization option
    if current_file_path:
        load_and_view_model(current_file_path)


def toggle_mode():

    global interactive_mode
    interactive_mode = not interactive_mode  # Switch between True and False
    if interactive_mode:
        print("Switched to Interactive Mode")
    else:
        print("Switched to Automatic Update Mode")

    # If a model is already loaded, re-render it to apply the mode change
    if current_file_path:
        load_and_view_model(current_file_path)


def on_model_select(event, tree):

    global current_file_path
    # Get the selected item
    selected_item = tree.selection()[0]
    file_path = tree.item(selected_item, 'values')

    # If a file is selected (it will have a file path as value)
    if file_path:
        current_file_path = file_path[0]
        load_and_view_model(current_file_path)


def initialize_visualizer():

    global vis
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Model Viewer", width=800, height=600)


def load_and_view_model(file_path):

    global vis, vis_option, interactive_mode
    if vis is None:
        initialize_visualizer()

    # Read the mesh
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Normalize the mesh (scale and center)
    mesh = normalize_mesh(mesh)

    # Clear previous geometries
    vis.clear_geometries()

    # Add the new mesh to the visualizer based on the selected option
    if vis_option == "smoothshade":
        vis.add_geometry(mesh)

    elif vis_option == "wireframe_on_shaded":
        vis.add_geometry(mesh)
        vis.get_render_option().mesh_show_wireframe = True

    elif vis_option == "wireframe":
        # Create wireframe using LineSet
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        vis.add_geometry(wireframe)

    elif vis_option == "world_axes":
        # Display the mesh with a world axis system.
        line_endpoints = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        line_indices = [[0, 1], [0, 2], [0, 3]]
        world_axes = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_endpoints),
            lines=o3d.utility.Vector2iVector(line_indices)
        )
        vis.add_geometry(mesh)
        vis.add_geometry(world_axes)

    elif vis_option == "black_background":
        vis.add_geometry(mesh)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])

    # Handle the mode change
    if interactive_mode:
        # Run the interactive visualizer where user can rotate, zoom, etc.
        vis.run()  # Block and let the user interact with the model
        vis.destroy_window()  # Close after interaction is complete
        vis = None  # Reset the visualizer for next usage
    else:
        # Automatic update mode (non-blocking)
        vis.poll_events()
        vis.update_renderer()


def normalize_mesh(mesh):

    # Compute the bounding box of the mesh
    bbox = mesh.get_axis_aligned_bounding_box()

    # Get the center of the bounding box
    center = bbox.get_center()

    # Translate the mesh to center it at the origin
    mesh.translate(-center)

    # Compute the scale factor to fit the mesh within a unit cube
    extent = bbox.get_extent()  # Use get_extent() to obtain the width, height, and depth
    scale_factor = 1.0 / max(extent)  # Scale by the largest dimension
    mesh.scale(scale_factor, center=[0, 0, 0])

    return mesh


def init_gui():
    global tree
    # Initialize the main window
    root = tk.Tk()
    root.title("3D Model Viewer")

    # Create a PanedWindow for the split layout
    paned_window = tk.PanedWindow(root, orient=tk.HORIZONTAL)
    paned_window.pack(fill=tk.BOTH, expand=1)

    # Create the left frame for the treeview (file explorer)
    left_frame = tk.Frame(paned_window)
    paned_window.add(left_frame)

    # Create Treeview widget for folder navigation
    tree = ttk.Treeview(left_frame)
    tree.pack(fill=tk.Y, expand=True)

    # Add a scroll bar to the Treeview
    scrollbar = tk.Scrollbar(left_frame, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill="y")

    # Bind the event for when an item in the tree is clicked
    tree.bind('<<TreeviewSelect>>', lambda event: on_model_select(event, tree))

    # Create the right frame for the Open3D viewer controls (buttons)
    right_frame = tk.Frame(paned_window)
    paned_window.add(right_frame)

    # Add buttons for visualization options
    create_buttons(right_frame)

    # Load the initial folder structure
    load_folder_structure(tree, "ShapeDatabase_INFOMR-master")  # Replace with your model folder path

    root.mainloop()


if __name__ == "__main__":
    init_gui()
