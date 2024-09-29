import open3d as o3d
import os
import subprocess


# Function to convert OBJ to OFF using Open3D
def obj_to_off_with_open3d(obj_filepath, off_filepath):
    # Load the mesh using Open3D
    mesh = o3d.io.read_triangle_mesh(obj_filepath)

    # Save the mesh in OFF format
    o3d.io.write_triangle_mesh(off_filepath, mesh)
    print(f"Converted: {obj_filepath} -> {off_filepath}")


# Function to process all .obj files in a folder and its subfolders
def convert_all_obj_to_off(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_filepath = os.path.join(root, file)

                print(f"Converting: {obj_filepath}")

                # Create the output .off filename (same name but with .off extension)
                off_filepath = os.path.splitext(obj_filepath)[0] + ".off"

                # Convert the OBJ file to OFF
                obj_to_off_with_open3d(obj_filepath, off_filepath)

                # Change NOFF to OFF if necessary
                replace_noff_with_off(off_filepath)

                # Perform subdivision on the OFF file
                subdivided_filepath = subdivide_with_doosabin(off_filepath, times=1)

                # Perform tessellation on the subdivided file
                if subdivided_filepath:
                    tessellated_filepath = tessellate_with_tess(subdivided_filepath)

                    # Convert the tessellated file back to OBJ format
                    if tessellated_filepath:
                        off_to_obj_with_open3d(tessellated_filepath, obj_filepath)

                # Clean up intermediate files
                cleanup_files([off_filepath, subdivided_filepath, tessellated_filepath])


# Function to replace NOFF with OFF in .off files
def replace_noff_with_off(off_filepath):
    with open(off_filepath, 'r') as file:
        lines = file.readlines()

    # Check if the first line is "NOFF" and change it to "OFF"
    if lines[0].strip() == "NOFF":
        lines[0] = "OFF\n"
        with open(off_filepath, 'w') as file:
            file.writelines(lines)
        print(f"Replaced NOFF with OFF in: {off_filepath}")


# Function to perform subdivision using doosabin.jar
def subdivide_with_doosabin(off_filepath, times=1):
    # Define the initial input file path
    current_filepath = off_filepath

    # Path to your doosabin.jar file (update this path if needed)
    doosabin_jar_path = "doosabin.jar"  # Update this path accordingly

    for i in range(times):
        # Define the output path for the subdivided file
        subdivided_filepath = os.path.splitext(current_filepath)[0] + f"_subdivided_{i + 1}.off"

        # Run the doosabin.jar command using subprocess
        try:
            subprocess.run(["java", "-jar", doosabin_jar_path, current_filepath, subdivided_filepath], check=True)
            print(f"Subdivision {i + 1} completed: {current_filepath} -> {subdivided_filepath}")
        except subprocess.CalledProcessError:
            print(f"Error during subdivision {i + 1} of: {current_filepath}")
            return current_filepath

        # Update the current file path for the next iteration
        current_filepath = subdivided_filepath

    return current_filepath


# Function to perform tessellation using tess.jar
def tessellate_with_tess(off_filepath):
    # Define the output path for the tessellated file
    tessellated_filepath = os.path.splitext(off_filepath)[0] + "_tessellated.off"

    # Path to your tess.jar file (update this path if needed)
    tess_jar_path = "tess.jar"  # Update this path accordingly

    # Run the tess.jar command using subprocess
    try:
        subprocess.run(["java", "-jar", tess_jar_path, off_filepath, tessellated_filepath], check=True)
        print(f"Tessellation completed: {off_filepath} -> {tessellated_filepath}")
        return tessellated_filepath
    except subprocess.CalledProcessError:
        print(f"Error during tessellation of: {off_filepath}")
        return None


# Function to convert OFF to OBJ using Open3D, keeping the original OBJ filename
def off_to_obj_with_open3d(off_filepath, original_obj_filepath):
    # Load the mesh using Open3D
    mesh = o3d.io.read_triangle_mesh(off_filepath)

    # Save the mesh in OBJ format using the original OBJ file name
    o3d.io.write_triangle_mesh(original_obj_filepath, mesh)
    print(f"Converted back to OBJ: {off_filepath} -> {original_obj_filepath}")


# Function to clean up intermediate files
def cleanup_files(filepaths):
    for filepath in filepaths:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"Deleted intermediate file: {filepath}")
            except OSError as e:
                print(f"Error deleting file {filepath}: {e}")


def display_vertices_and_faces(refined_folder):
    # Traverse through all files in the refined folder and its subfolders
    for root, dirs, files in os.walk(refined_folder):
        for file in files:
            # Check if the file is an OBJ model
            if file.lower().endswith('.obj'):
                file_path = os.path.join(root, file)

                # Load the mesh using Open3D
                mesh = o3d.io.read_triangle_mesh(file_path)

                # Get the number of vertices and faces
                num_vertices = len(mesh.vertices)
                num_faces = len(mesh.triangles)

                # Display the results
                print(f"Model: {file_path}")
                print(f" - Number of vertices: {num_vertices}")
                print(f" - Number of faces: {num_faces}")
                print("-" * 40)


if __name__ == "__main__":
    # Define the path to your database folder containing OBJ models
    database_folder = "RefineNeeded"  # Change this path if your folder is in a different location

    # Start the conversion, subdivision, tessellation, and conversion back to OBJ process
    convert_all_obj_to_off(database_folder)

    # Display vertices and faces information
    #display_vertices_and_faces(database_folder)

    print("Conversion, subdivision, tessellation, and OBJ export process completed!")
