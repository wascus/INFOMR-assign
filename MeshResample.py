import os
import open3d as o3d
import shutil
import pandas as pd

# Read the data from Excel to get bin sizes
df = pd.read_excel('shape_analysis_final.xlsx')

# Calculate statistics to get the min and max values
min_faces = df['num_faces'].min()
max_faces = df['num_faces'].max()

# Define the number of bins (as used in the histogram)
num_bins = 20

# Calculate bin size for faces
face_bin_size = (max_faces - min_faces) / num_bins

# Define the formats to check
valid_mesh_formats = ['.ply', '.stl', '.obj', '.off']

def is_poorly_sampled(num_faces, max_faces=1000):

    return num_faces < max_faces

def is_in_face_range(num_faces, min_faces, max_faces):

    return min_faces <= num_faces <= max_faces

def is_heavily_sampled_by_vertices(num_faces, num_vertices, min_faces=5000):

    return num_vertices > min_faces

def check_mesh(file_path):

    try:
        # Load the mesh using Open3D
        mesh = o3d.io.read_triangle_mesh(file_path)

        # Check if the mesh has normals; if not, compute them
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        # Return the number of faces (triangles) and vertices in the mesh
        return len(mesh.triangles), len(mesh.vertices)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def copy_mesh_to_folder(file_path, root, base_folder, destination_folder):

    relative_path = os.path.relpath(root, base_folder)  # Keep the relative subfolder path
    dest_folder = os.path.join(destination_folder, relative_path)
    os.makedirs(dest_folder, exist_ok=True)  # Create the folder if it doesn't exist
    shutil.copy(file_path, dest_folder)  # Copy the file

def process_folder(base_folder, refined_folder, heavily_sampled_folder, ranges_folder, num_bins=20):

    total_meshes = 0
    meshes_needing_refinement = 0
    meshes_heavily_sampled = 0
    meshes_in_range = {}

    # Traverse through all meshes in the base folder and its subfolders
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_mesh_formats):
                file_path = os.path.join(root, file)
                total_meshes += 1
                num_faces, num_vertices = check_mesh(file_path)

                if num_faces is None or num_vertices is None:
                    continue  # Skip files that failed to load

                # If mesh has fewer than 1000 faces and vertices, it goes to RefineNeeded
                if is_poorly_sampled(num_faces=num_faces, max_faces=6500):
                    print(f"Mesh {file_path} needs refinement.")
                    copy_mesh_to_folder(file_path, root, base_folder, refined_folder)
                    meshes_needing_refinement += 1
                # If mesh has more than 5000 vertices, it goes to the HeavilySampled folder
                elif is_heavily_sampled_by_vertices(num_faces, num_vertices, min_faces=15000):
                    print(f"Mesh {file_path} is heavily sampled (vertices > 5000).")
                    copy_mesh_to_folder(file_path, root, base_folder, heavily_sampled_folder)
                    meshes_heavily_sampled += 1
                # Otherwise, categorize the mesh into one of the range folders based on histogram bins
                else:
                    for i in range(num_bins):
                        min_faces_bin = min_faces + i * face_bin_size
                        max_faces_bin = min_faces_bin + face_bin_size - 1  # Prevent overlap
                        if is_in_face_range(num_faces, min_faces_bin, max_faces_bin):
                            folder_name = f"Range_{int(min_faces_bin)}_{int(max_faces_bin)}"
                            if folder_name not in meshes_in_range:
                                meshes_in_range[folder_name] = 0
                            meshes_in_range[folder_name] += 1
                            range_folder_path = os.path.join(ranges_folder, folder_name)
                            print(f"Mesh {file_path} is in the face range [{int(min_faces_bin)}, {int(max_faces_bin)}].")
                            copy_mesh_to_folder(file_path, root, base_folder, range_folder_path)
                            break

    # Print the summary of results
    print("\nSummary:")
    print(f"Total meshes checked: {total_meshes}")
    print(f"Meshes needing refinement: {meshes_needing_refinement}")
    print(f"Meshes heavily sampled: {meshes_heavily_sampled}")
    print(f"Meshes in defined face ranges:")
    for folder_name, count in meshes_in_range.items():
        print(f"{folder_name}: {count} meshes")

if __name__ == "__main__":
    # Base folder path (Replace this with the path to your ShapeDatabase_INFOMR-master folder)
    base_folder = "ShapeDatabase_INFOMR-master"  # Replace with the actual path to your folder

    # Refined folder where the refined meshes (faces < 1000 and vertices < 1000) will be copied
    refined_folder = "RefineNeeded"

    # HeavilySampled folder where the heavily sampled meshes (vertices > 5000) will be copied
    heavily_sampled_folder = "HeavilySampled"

    # Ranges folder where the meshes in face ranges will be copied
    ranges_folder = "RangeFolders"

    # Start processing the base folder, retaining the category structure
    process_folder(base_folder, refined_folder, heavily_sampled_folder, ranges_folder)

    print("\nProcessing completed!")
