import os
import pymeshlab
import random

# Function to process a percentage of .obj files in a folder and its subfolders
def process_percentage_of_obj_files_with_subsampling(root_folder, mlx_file, target_face_count, percentage):
    obj_files = []

    # Traverse the folder and subfolders to collect all .obj files
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_filepath = os.path.join(root, file)
                obj_files.append(obj_filepath)

    # If there are files, select the specified percentage of them
    if obj_files:
        num_files_to_process = int(len(obj_files) * (percentage / 100))  # Calculate the number of files based on percentage
        selected_files = random.sample(obj_files, num_files_to_process)  # Randomly select files based on the percentage

        print(f"Processing {num_files_to_process} out of {len(obj_files)} .obj files ({percentage}% of total).")

        # Process each selected .obj file
        for obj_filepath in selected_files:
            print(f"Processing: {obj_filepath}")

            # Perform subsampling using PyMeshLab and save with the same name
            success = subsample_with_pymeshlab(obj_filepath, mlx_file, target_face_count)

            if success:
                print(f"Subsampling completed and saved for: {obj_filepath}")
            else:
                print(f"Skipping file due to an error: {obj_filepath}")

# Function to perform subsampling iteratively using PyMeshLab
def subsample_with_pymeshlab(obj_filepath, mlx_file, target_face_count):
    try:
        # Initialize the PyMeshLab MeshSet
        ms = pymeshlab.MeshSet()

        # Load the OBJ file into the MeshSet
        ms.load_new_mesh(obj_filepath)

        # Check the initial number of faces
        current_face_count = ms.current_mesh().face_number()
        print(f"Initial face count: {current_face_count}")

        # Loop until the mesh reaches or goes below the desired face count
        iteration = 0
        while current_face_count > target_face_count:
            iteration += 1
            print(f"Iteration {iteration}: Applying filter to reduce face count...")

            # Load and apply the filter script from the mlx file for subsampling
            ms.load_filter_script(mlx_file)
            ms.apply_filter_script()

            # Check the number of faces after applying the filter
            current_face_count = ms.current_mesh().face_number()
            print(f"Current face count after applying filter: {current_face_count}")


            # Break out if the target face count is reached or if there are no more faces
            if current_face_count <= target_face_count or current_face_count == 0:
                print(f"Stopping iteration. Final face count: {current_face_count}")
                break

        # Save the subsampled mesh back to the original file path, replacing the original
        ms.save_current_mesh(obj_filepath, save_polygonal=False)
        print(f"Subsampling completed using PyMeshLab and saved as: {obj_filepath} with {current_face_count} faces.")

        return True

    except Exception as e:
        print(f"Error during subsampling with PyMeshLab for {obj_filepath}: {e}")
        return False  # Return False to indicate failure

if __name__ == "__main__":
    # Define the path to your database folder containing OBJ models
    database_folder = "HeavilySampled"  # Change this path if your folder is in a different location

    # Define the path to your MLX file for subsampling
    mlx_file_path = "Subsampling.mlx"  # Change this path if your MLX file is in a different location

    # Define your target face count
    target_face_count = 13000  # Set this to your desired number of faces

    # Define the percentage of .obj files to process
    percentage_to_process = 100  # Adjust this percentage as needed

    # Start the subsampling process using PyMeshLab, processing the specified percentage of the OBJ files
    process_percentage_of_obj_files_with_subsampling(database_folder, mlx_file_path, target_face_count, percentage_to_process)

    print(f"Subsampling with PyMeshLab and OBJ export process for {percentage_to_process}% of files completed!")
