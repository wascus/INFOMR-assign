import pymeshlab
import tempfile
import os
import open3d as o3d

def clean_open3d_mesh_with_disk(o3d_mesh, mlx_file):
    """
    Perform a single pass of a cleaning operation on an Open3D mesh using PyMeshLab and return the processed mesh.
    This version saves the mesh to disk to run the MLX filter.

    :param o3d_mesh: Open3D mesh object to process.
    :param mlx_file: Path to the MLX file for cleaning operations.
    :return: A new Open3D mesh object with the cleaning operation applied.
    """
    try:
        # Create a temporary directory to save the input and output files
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "temp_input.obj")
            output_file = os.path.join(temp_dir, "temp_output.obj")

            # Save the Open3D mesh to a temporary OBJ file
            o3d.io.write_triangle_mesh(input_file, o3d_mesh)

            # Initialize the PyMeshLab MeshSet
            ms = pymeshlab.MeshSet()

            # Load the temporary OBJ file into the MeshSet
            ms.load_new_mesh(input_file)

            print("Applying cleaning filter...")

            # Load and apply the filter script from the MLX file
            ms.load_filter_script(mlx_file)
            ms.apply_filter_script()

            # Save the cleaned mesh to the output file
            ms.save_current_mesh(output_file, save_polygonal=False)
            print("Cleaning operation completed and saved as temporary file.")

            # Load the processed OBJ back into an Open3D mesh
            processed_mesh = o3d.io.read_triangle_mesh(output_file)
            return processed_mesh

    except Exception as e:
        print(f"Error during cleaning operation with PyMeshLab: {e}")
        return None

# Example function to integrate this into another script
def Clean(o3d_mesh):
    # Define the path to your MLX file
    mlx_file_path = "Clean.mlx"  # Replace with the actual path to your MLX file

    # Perform cleaning operation and return the new Open3D mesh
    return clean_open3d_mesh_with_disk(o3d_mesh, mlx_file_path)
