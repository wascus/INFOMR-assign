import pymeshlab
import tempfile
import open3d as o3d
import os


def supersample_with_threshold(o3d_mesh, mlx_file, target_face_count=10000, min_face_count=8000, initial_threshold=5, max_threshold=30, threshold_step=0.005):
    """
    Perform supersampling on an Open3D mesh using PyMeshLab with threshold-based iterative logic
    and ensure the face count is within the desired range.

    :param o3d_mesh: Open3D mesh object to process.
    :param mlx_file: Path to the MLX file for additional filters.
    :param target_face_count: Target maximum number of faces for the supersampled mesh.
    :param min_face_count: Minimum acceptable number of faces for the supersampled mesh.
    :param initial_threshold: Starting threshold value for subdivision.
    :param max_threshold: Maximum threshold value to stop trying subdivision.
    :param threshold_step: Increment for adjusting the threshold.
    :return: A new Open3D mesh object with the supersampled geometry, or None if processing failed.
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

            # Initialize threshold
            threshold = initial_threshold
            face_count_achieved = False

            # Iteratively adjust the threshold and apply subdivision
            while threshold <= max_threshold:
                print(f"Processing with threshold: {threshold}")

                # Load and apply the filter script from the MLX file
                ms.load_filter_script(mlx_file)
                ms.apply_filter_script()

                # Apply subdivision with the current threshold
                ms.meshing_surface_subdivision_midpoint(iterations=1, threshold=pymeshlab.PercentageValue(threshold))

                # Check the number of faces after processing
                new_face_count = ms.current_mesh().face_number()
                print(f"Face count after subdivision with threshold {threshold}: {new_face_count}")

                # If the face count is within the acceptable range, exit the loop
                if min_face_count <= new_face_count <= target_face_count:
                    face_count_achieved = True
                    print(f"Face count is within the acceptable range: {new_face_count}")
                    break

                # Increment threshold and retry
                print(f"Face count {new_face_count} is outside the range. Adjusting threshold...")
                threshold += threshold_step

            # Final check if the face count is below min_face_count
            if not face_count_achieved:
                print(f"Face count is below {min_face_count}. Reapplying filters until it exceeds {min_face_count}.")

                while new_face_count < min_face_count:
                    ms.load_filter_script(mlx_file)
                    ms.apply_filter_script()

                    # Check the updated face count
                    new_face_count = ms.current_mesh().face_number()
                    print(f"Reapplying filter. Current face count: {new_face_count}")

                    # Break to avoid infinite loops if the face count stops increasing
                    if new_face_count >= min_face_count or new_face_count == 0:
                        break

            # Save the final processed mesh to the output file
            ms.save_current_mesh(output_file, save_polygonal=False)

            # Load the processed OBJ back into an Open3D mesh
            processed_mesh = o3d.io.read_triangle_mesh(output_file)
            return processed_mesh

    except Exception as e:
        print(f"Error during supersampling with PyMeshLab: {e}")
        return None


# Example function to integrate into another script
def Supersample(o3d_mesh):
    # Define the path to your MLX file
    mlx_file_path = "Supersampling.mlx"  # Replace with the actual path to your MLX file

    # Perform supersampling and return the new Open3D mesh
    return supersample_with_threshold(o3d_mesh, mlx_file_path)
