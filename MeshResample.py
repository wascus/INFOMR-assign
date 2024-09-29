import os
import open3d as o3d
import shutil

# Define the formats to check
valid_mesh_formats = ['.ply', '.stl', '.obj', '.off']

def is_poorly_sampled(mesh):
    """
    Checks if a mesh is poorly sampled.
    A mesh is poorly sampled if it has fewer than 100 vertices or faces.
    """
    num_vertices = len(mesh.vertices)
    num_faces = len(mesh.triangles)
    return num_vertices < 300 or num_faces < 300

def is_heavily_sampled(mesh):
    """
    Checks if a mesh has more than 7500 vertices.
    """
    return len(mesh.vertices) > 7500 or len(mesh.triangles) > 7500

def check_mesh(file_path):
    """
    Loads a mesh from the given file path and checks if it is poorly sampled or heavily sampled.
    Returns True if the mesh needs refinement, False if it's heavily sampled, and None otherwise.
    """
    try:
        # Load the mesh using Open3D
        mesh = o3d.io.read_triangle_mesh(file_path)

        # Check if the mesh has normals; if not, compute them
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        # Check if the mesh is poorly sampled
        if is_poorly_sampled(mesh):
            return True
        # Check if the mesh is heavily sampled
        elif is_heavily_sampled(mesh):
            return False
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return None

def copy_mesh_to_refine_folder(file_path, root, base_folder, refine_folder):
    """
    Copies the mesh that needs refinement into the RefineNeeded folder, maintaining subfolder structure.
    """
    relative_path = os.path.relpath(root, base_folder)
    dest_folder = os.path.join(refine_folder, relative_path)
    os.makedirs(dest_folder, exist_ok=True)
    shutil.copy(file_path, dest_folder)

def copy_mesh_to_heavily_sampled_folder(file_path, root, base_folder, heavily_sampled_folder):
    """
    Copies the mesh that is heavily sampled into the HeavilySampled folder, maintaining subfolder structure.
    """
    relative_path = os.path.relpath(root, base_folder)
    dest_folder = os.path.join(heavily_sampled_folder, relative_path)
    os.makedirs(dest_folder, exist_ok=True)
    shutil.copy(file_path, dest_folder)

def process_folder(base_folder, refine_folder, heavily_sampled_folder):
    """
    Traverses the base folder and all subfolders to check for poorly sampled or heavily sampled meshes.
    Copies meshes needing refinement to the RefineNeeded folder and heavily sampled meshes to HeavilySampled folder.
    Prints the names of meshes that need refinement and the total count.
    """
    total_meshes = 0
    meshes_needing_refinement = 0
    meshes_heavily_sampled = 0

    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_mesh_formats):
                file_path = os.path.join(root, file)
                total_meshes += 1
                check_result = check_mesh(file_path)

                if check_result is True:
                    print(f"Mesh {file_path} needs refinement.")
                    copy_mesh_to_refine_folder(file_path, root, base_folder, refine_folder)
                    meshes_needing_refinement += 1
                elif check_result is False:
                    print(f"Mesh {file_path} is heavily sampled.")
                    copy_mesh_to_heavily_sampled_folder(file_path, root, base_folder, heavily_sampled_folder)
                    meshes_heavily_sampled += 1

    print("\nSummary:")
    print(f"Total meshes checked: {total_meshes}")
    print(f"Meshes needing refinement: {meshes_needing_refinement}")
    print(f"Meshes heavily sampled: {meshes_heavily_sampled}")

if __name__ == "__main__":
    # Base folder path (Replace this with the path to your ShapeDatabase_INFOMR-master folder)
    base_folder = "ShapeDatabase_INFOMR-master"  # Replace with the actual path to your folder

    # RefineNeeded folder where the refined meshes will be copied
    refine_folder = "RefineNeeded"  # The output folder for meshes needing refinement

    # HeavilySampled folder where the heavily sampled meshes will be copied
    heavily_sampled_folder = "HeavilySampled"  # The output folder for heavily sampled meshes

    # Start processing the base folder
    process_folder(base_folder, refine_folder, heavily_sampled_folder)

    print("\nProcessing completed!")
