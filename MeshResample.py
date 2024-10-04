import os
import open3d as o3d
import shutil
import random

# Define the formats to check
valid_mesh_formats = ['.ply', '.stl', '.obj', '.off']

def is_poorly_sampled(mesh):
    """
    Checks if a mesh is poorly sampled.
    A mesh is poorly sampled if it has fewer than 300 vertices or faces.
    """
    num_vertices = len(mesh.vertices)
    num_faces = len(mesh.triangles)
    return num_vertices < 300 or num_faces < 300

def is_heavily_sampled(mesh):
    """
    Checks if a mesh has more than 7500 vertices or faces.
    """
    return len(mesh.vertices) > 7500 or len(mesh.triangles) > 7500

def is_in_face_range(mesh, min_faces=5000, max_faces=6000):
    """
    Checks if a mesh has a number of faces in the given range [min_faces, max_faces].
    """
    num_faces = len(mesh.triangles)
    return min_faces <= num_faces <= max_faces

def check_mesh(file_path, min_faces=5000, max_faces=6000):
    """
    Loads a mesh from the given file path and checks if it is poorly sampled, heavily sampled, or within a given face range.
    Returns:
        - 'refine' if the mesh needs refinement (poorly sampled),
        - 'heavily_sampled' if the mesh is heavily sampled,
        - 'in_range' if the mesh has faces within the range [min_faces, max_faces],
        - None otherwise.
    """
    try:
        # Load the mesh using Open3D
        mesh = o3d.io.read_triangle_mesh(file_path)

        # Check if the mesh has normals; if not, compute them
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        # Check if the mesh is poorly sampled
        if is_poorly_sampled(mesh):
            return 'refine'
        # Check if the mesh is heavily sampled
        elif is_heavily_sampled(mesh):
            return 'heavily_sampled'
        # Check if the mesh has faces within the desired range
        elif is_in_face_range(mesh, min_faces, max_faces):
            return 'in_range'
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return None

def copy_mesh_to_folder(file_path, root, base_folder, destination_folder):
    """
    Copies the mesh into the specified destination folder, maintaining the subfolder structure.
    """
    relative_path = os.path.relpath(root, base_folder)
    dest_folder = os.path.join(destination_folder, relative_path)
    os.makedirs(dest_folder, exist_ok=True)
    shutil.copy(file_path, dest_folder)

def process_folder(base_folder, refine_folder, heavily_sampled_folder, range_folder, min_faces=5000, max_faces=6000, percentage=50):
    """
    Traverses the base folder and all subfolders to check for poorly sampled, heavily sampled meshes, or meshes within the given face range.
    Copies meshes needing refinement to the RefineNeeded folder, heavily sampled meshes to HeavilySampled folder, and in-range meshes to RangeSelected folder.
    Randomly selects a user-specified percentage of the in-range meshes for copying to the RangeSelected folder.
    Prints the names of meshes that need refinement and the total count.
    """
    total_meshes = 0
    meshes_needing_refinement = 0
    meshes_heavily_sampled = 0
    meshes_in_range = []

    # Traverse through all meshes
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_mesh_formats):
                file_path = os.path.join(root, file)
                total_meshes += 1
                check_result = check_mesh(file_path, min_faces, max_faces)

                if check_result == 'refine':
                    print(f"Mesh {file_path} needs refinement.")
                    copy_mesh_to_folder(file_path, root, base_folder, refine_folder)
                    meshes_needing_refinement += 1
                elif check_result == 'heavily_sampled':
                    print(f"Mesh {file_path} is heavily sampled.")
                    copy_mesh_to_folder(file_path, root, base_folder, heavily_sampled_folder)
                    meshes_heavily_sampled += 1
                elif check_result == 'in_range':
                    print(f"Mesh {file_path} is in the face range [{min_faces}, {max_faces}].")
                    meshes_in_range.append((file_path, root))  # Collect in-range meshes

    # Calculate the number of meshes to select based on the percentage
    num_to_select = max(1, int((percentage / 100) * len(meshes_in_range)))  # Ensure at least 1 is selected
    selected_meshes = random.sample(meshes_in_range, k=num_to_select)

    # Copy the selected meshes to the RangeSelected folder
    for file_path, root in selected_meshes:
        copy_mesh_to_folder(file_path, root, base_folder, range_folder)

    print("\nSummary:")
    print(f"Total meshes checked: {total_meshes}")
    print(f"Meshes needing refinement: {meshes_needing_refinement}")
    print(f"Meshes heavily sampled: {meshes_heavily_sampled}")
    print(f"Meshes in face range [{min_faces}, {max_faces}]: {len(meshes_in_range)}")
    print(f"Randomly selected {percentage}% of in-range meshes: {len(selected_meshes)}")

if __name__ == "__main__":
    # Base folder path (Replace this with the path to your ShapeDatabase_INFOMR-master folder)
    base_folder = "ShapeDatabase_INFOMR-master"  # Replace with the actual path to your folder

    # RefineNeeded folder where the refined meshes will be copied
    refine_folder = "RefineNeeded"  # The output folder for meshes needing refinement

    # HeavilySampled folder where the heavily sampled meshes will be copied
    heavily_sampled_folder = "HeavilySampled"  # The output folder for heavily sampled meshes

    # RangeSelected folder where the meshes in the specified range will be copied
    range_folder = "RangeSelected"  # The output folder for meshes within the face range

    # Define the percentage of meshes to select (customizable by the user)
    selection_percentage = 30  # Replace with the desired percentage (e.g., 50)

    # Start processing the base folder, specifying the face range (5k to 6k faces)
    process_folder(base_folder, refine_folder, heavily_sampled_folder, range_folder, min_faces=5000, max_faces=5900, percentage=selection_percentage)

    print("\nProcessing completed!")
