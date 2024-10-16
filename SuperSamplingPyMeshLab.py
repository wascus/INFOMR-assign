import os
import shutil
import pymeshlab


# Function to process all .obj files in a folder and its subfolders
def process_all_obj_files(root_folder, max_faces=10000):
    count = 0
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_filepath = os.path.join(root, file)
                print(f"Processing: {obj_filepath}")

                # Perform subdivision and additional filters using PyMeshLab
                success = process_mesh_with_pymeshlab(obj_filepath, max_faces)

                if success:
                    print(f"Processing completed and saved for: {obj_filepath}")
                    count += 1
                else:
                    print(f"File moved to 'RefineAgain' due to exceeding face limit: {obj_filepath}")

    print(f"Total successfully processed files: {count}")


# Function to apply subdivision and additional filters using PyMeshLab
def process_mesh_with_pymeshlab(obj_filepath, max_faces):
    try:

        # Initialize the threshold for subdivision
        threshold = 5
        max_threshold = 30  # Define a minimum threshold value to stop trying to subdivide

        while threshold <= max_threshold:
            # Initialize the PyMeshLab MeshSet
            ms = pymeshlab.MeshSet()

            # Load the original OBJ file into the MeshSet
            ms.load_new_mesh(obj_filepath)

            ms.load_filter_script('Subdivisions.mlx')
            ms.apply_filter_script()
            # Apply subdivision with the current threshold

            threshold_midpoint=pymeshlab.PercentageValue(threshold)

            ms.meshing_surface_subdivision_midpoint(iterations=1, threshold=threshold_midpoint)
            # Check the face count after applying the filters
            new_face_count = ms.current_mesh().face_number()
            print(f"New face count after processing with threshold {threshold}: {new_face_count}")

            # If the new face count is within the max_faces limit, save the mesh and break the loop
            if new_face_count <= max_faces:
                ms.save_current_mesh(obj_filepath, save_polygonal=False)
                print(f"Processing successful with {new_face_count} faces. Original file overwritten.")
                return True

            # If the face count exceeds the limit, reduce the threshold and try again
            print(f"Face count {new_face_count} exceeds {max_faces}. Lowering threshold and retrying...")
            threshold += 0.005

        # If we exit the loop without getting under the face limit, move the file to RefineAgain
        # move_to_refine_again_folder(obj_filepath, os.path.dirname(obj_filepath))
    ##print(f"Face count could not be reduced below {max_faces}. File moved to RefineAgain.")
        return False

    except Exception as e:
        print(f"Error during processing with PyMeshLab for {obj_filepath}: {e}")
        return False  # Return False to indicate failure


# Function to move files exceeding face count limit to 'RefineAgain' folder
def move_to_refine_again_folder(original_filepath, root_folder):
    # Define the RefineAgain folder path
    refine_again_folder = "RefineAgain"

    # Get the relative path from the root folder to the file
    relative_path = os.path.relpath(original_filepath, root_folder)

    # Create the full destination path, preserving the folder structure
    destination_filepath = os.path.join(refine_again_folder, relative_path)

    # Ensure the destination folder exists
    os.makedirs(os.path.dirname(destination_filepath), exist_ok=True)

    # Copy the original file to the new location
    shutil.copy2(original_filepath, destination_filepath)

    print(f"Original file copied to: {destination_filepath}")


if __name__ == "__main__":
    # Define the path to your database folder containing OBJ models
    database_folder = "RefineNeeded2"  # Change this path if your folder is in a different location

    # Start the subdivision process using PyMeshLab
    process_all_obj_files(database_folder)

    print("Processing with PyMeshLab and OBJ export process completed!")
