import pymeshlab
import tempfile
import os
import open3d as o3d

def subsample_open3d_mesh(o3d_mesh, mlx_file, target_face_count):
    """
    Perform subsampling on an Open3D mesh using PyMeshLab and return the processed Open3D mesh.

    :param o3d_mesh: Open3D mesh object to process.
    :param mlx_file: Path to the MLX file for subsampling.
    :param target_face_count: Desired number of faces in the subsampled mesh.
    :return: A new Open3D mesh object with the subsampled geometry.
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

            # Check the initial number of faces
            current_face_count = ms.current_mesh().face_number()
            print(f"Initial face count: {current_face_count}")

            # Loop until the mesh reaches or goes below the desired face count
            iteration = 0
            while current_face_count > target_face_count:
                iteration += 1
                print(f"Iteration {iteration}: Applying filter to reduce face count...")

                # Load and apply the filter script from the MLX file for subsampling
                ms.load_filter_script(mlx_file)
                ms.apply_filter_script()

                # Check the number of faces after applying the filter
                current_face_count = ms.current_mesh().face_number()
                print(f"Current face count after applying filter: {current_face_count}")

                # Break out if the target face count is reached or if there are no more faces
                if current_face_count <= target_face_count or current_face_count == 0:
                    print(f"Stopping iteration. Final face count: {current_face_count}")
                    break

            # Save the subsampled mesh to the output file
            ms.save_current_mesh(output_file, save_polygonal=False)
            print(f"Subsampling completed and saved as temporary file.")

            # Load the processed OBJ back into an Open3D mesh
            processed_mesh = o3d.io.read_triangle_mesh(output_file)
            return processed_mesh

    except Exception as e:
        print(f"Error during subsampling with PyMeshLab: {e}")
        return None

# Example function to integrate this into another script
def Subsample(o3d_mesh):
    # Define the path to your MLX file
    mlx_file_path = "Subsampling.mlx"  # Replace with the actual path to your MLX file

    # Define the target face count
    target_face_count = 13000

    # Perform subsampling and return the new Open3D mesh
    return subsample_open3d_mesh(o3d_mesh, mlx_file_path, target_face_count)
