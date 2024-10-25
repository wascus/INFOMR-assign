import open3d as o3d
import os
import subprocess
import concurrent.futures

# Function to convert OBJ to OFF using Open3D
def obj_to_off_with_open3d(obj_filepath, off_filepath):
    # Load the mesh using Open3D
    mesh = o3d.io.read_triangle_mesh(obj_filepath)

    # Save the mesh in OFF format
    o3d.io.write_triangle_mesh(off_filepath, mesh)
    print(f"Converted: {obj_filepath} -> {off_filepath}")


# Function to check the vertex count of an OFF file
def get_vertex_count(off_filepath):
    # Load the mesh using Open3D
    mesh = o3d.io.read_triangle_mesh(off_filepath)
    return len(mesh.vertices)


# Function to run cleanoff.jar to reduce the number of vertices until it reaches 7500 or fewer
def reduce_vertices_with_cleanoff(off_filepath):
    # Path to your cleanoff.jar file (update this path if needed)
    cleanoff_jar_path = "cleanoff.jar"  # Update this path with the actual path of cleanoff.jar
    temp_off_filepath = os.path.splitext(off_filepath)[0] + "_temp.off"

    epsilon = 0.000512  # Start with a small epsilon value as a float
    max_epsilon = 100  # Define a maximum epsilon value to prevent excessive merging

    while True:
        vertex_count = get_vertex_count(off_filepath)
        print(f"Current vertex count for {off_filepath}: {vertex_count}")

        # If the vertex count is 7500 or fewer, break out of the loop
        if vertex_count <= 15000:
            print(f"Vertex count is now under the limit for {off_filepath}.")
            break

        # Run cleanoff.jar to reduce the vertices
        try:
            subprocess.run(
                ["java", "-jar", cleanoff_jar_path, off_filepath, temp_off_filepath, str(epsilon)],
                check=True, capture_output=True, text=True
            )
            print(f"Ran cleanoff.jar on {off_filepath} with epsilon={epsilon}")

            # Replace the original OFF file with the reduced version
            os.replace(temp_off_filepath, off_filepath)

            # Increase epsilon gradually to be more aggressive if needed
            epsilon = min(epsilon * 1.3, max_epsilon)  # Increase epsilon as a float

        except subprocess.CalledProcessError as e:
            print(f"Error running cleanoff.jar on {off_filepath}: {e}")
            print(f"Standard Output: {e.stdout}")
            print(f"Standard Error: {e.stderr}")
            break

    # Clean up the temporary file if it still exists
    if os.path.exists(temp_off_filepath):
        os.remove(temp_off_filepath)


# Function to convert OFF to OBJ using Open3D and rename to match the original file
def off_to_obj_with_open3d(off_filepath, final_obj_filepath):
    # Load the mesh using Open3D
    mesh = o3d.io.read_triangle_mesh(off_filepath)

    # Save the mesh in OBJ format, using the original OBJ file name
    o3d.io.write_triangle_mesh(final_obj_filepath, mesh)
    print(f"Converted back to OBJ: {off_filepath} -> {final_obj_filepath}")


# Function to clean up intermediate files
def cleanup_files(filepaths):
    for filepath in filepaths:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"Deleted intermediate file: {filepath}")
            except OSError as e:
                print(f"Error deleting file {filepath}: {e}")


# Function to handle processing of a single OBJ file
def process_single_obj_file(obj_filepath):
    try:
        # Create the output .off filename (same name but with .off extension)
        off_filepath = os.path.splitext(obj_filepath)[0] + ".off"

        print(f"Processing heavily sampled model: {obj_filepath}")

        # Convert the OBJ file to OFF
        obj_to_off_with_open3d(obj_filepath, off_filepath)

        # Reduce the vertices using cleanoff.jar until there are 7500 or fewer
        reduce_vertices_with_cleanoff(off_filepath)

        # Convert the reduced OFF file back to OBJ format, retaining the original OBJ filename
        off_to_obj_with_open3d(off_filepath, obj_filepath)

        # Clean up the intermediate OFF file
        cleanup_files([off_filepath])

    except Exception as e:
        print(f"Error processing file {obj_filepath}: {e}")


# Function to process all OBJ files in the HeavilySampled folder in parallel
def process_heavily_sampled_models_parallel(root_folder):
    obj_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_files.append(os.path.join(root, file))

    # Use a process pool to process files in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_single_obj_file, obj_files)


if __name__ == "__main__":
    # Define the path to your HeavilySampled folder containing OBJ models
    heavily_sampled_folder = "HeavilySampled"  # Change this path if your folder is in a different location

    # Start processing the HeavilySampled folder in parallel
    process_heavily_sampled_models_parallel(heavily_sampled_folder)

    print("Processing of heavily sampled models completed!")
