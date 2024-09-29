import os
import pymeshlab


# Function to process all .obj files in a folder and its subfolders
def process_all_obj_files(root_folder, mlx_file):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_filepath = os.path.join(root, file)

                print(f"Processing: {obj_filepath}")

                # Perform subdivision using PyMeshLab and save with the same name
                success = subdivide_with_pymeshlab(obj_filepath, mlx_file)

                if success:
                    print(f"Subdivision completed and saved for: {obj_filepath}")
                else:
                    print(f"Skipping file due to an error: {obj_filepath}")


# Function to perform subdivision using PyMeshLab
def subdivide_with_pymeshlab(obj_filepath, mlx_file):
    try:
        # Initialize the PyMeshLab MeshSet
        ms = pymeshlab.MeshSet()

        # Load the OBJ file into the MeshSet
        ms.load_new_mesh(obj_filepath)

        # Load and apply the filter script from the mlx file
        ms.load_filter_script(mlx_file)
        ms.apply_filter_script()

        # Save the subdivided mesh back to the original file path
        ms.save_current_mesh(obj_filepath, save_polygonal = False)
        print(f"Subdivision completed using PyMeshLab and saved as: {obj_filepath}")

        return True

    except Exception as e:
        print(f"Error during subdivision with PyMeshLab for {obj_filepath}: {e}")
        return False  # Return False to indicate failure


if __name__ == "__main__":
    # Define the path to your database folder containing OBJ models
    database_folder = "RefineNeeded"  # Change this path if your folder is in a different location

    # Define the path to your MLX file
    mlx_file_path = "Subdivisions.mlx"  # Change this path if your MLX file is in a different location

    # Start the subdivision process using PyMeshLab
    process_all_obj_files(database_folder, mlx_file_path)

    print("Subdivision with PyMeshLab and OBJ export process completed!")
