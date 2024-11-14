import os
import trimesh
import pandas as pd


#function to count the number of triangles and quads in each shape
def face_type(faces):
    face_types = {
        "triangles": 0,
        "quads": 0,
    }
    for face in faces:  #type of face determined based on the number of vertices
        num_vertices = len(face)
        if num_vertices == 3:
            face_types["triangles"] += 1
        elif num_vertices == 4:
            face_types["quads"] += 1
    return face_types


#function to analyze a shape
def analyze_shape(file_path, shape_class):
    try:
        file_name = os.path.splitext(os.path.basename(file_path))[0]  #get the object name
        mesh = trimesh.load_mesh(file_path)  #load the file as a trimesh object
        num_faces = len(mesh.faces)  #count the number of faces
        num_vertices = len(mesh.vertices)  #count the number of vertices
        face_type_counts = face_type(mesh.faces)  #count the types of faces
        bounding_box = mesh.bounding_box_oriented.vertices  #bounding box

        return {
            "shape": file_name,
            "class": shape_class,
            "num_faces": num_faces,
            "num_vertices": num_vertices,
            "triangles": face_type_counts["triangles"],
            "quads": face_type_counts["quads"],
            "bounding_box": bounding_box
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


#function to analyze all of the shapes in the database
def analyze_database(database_directory):
    results = []
    for shape_class in os.listdir(database_directory):
        class_directory = os.path.join(database_directory, shape_class)
        if os.path.isdir(class_directory):
            for file_name in os.listdir(class_directory):
                file_path = os.path.join(class_directory, file_name)
                if file_path.endswith('.obj'):
                    result = analyze_shape(file_path, shape_class)
                    if result:
                        results.append(result)
    return results


database_directory = "ShapeDatabase_INFOMR-master"  # path to directory !!!REPLACE!!!

# run the analysis on the database and save the results to an Excel file
results = pd.DataFrame(analyze_database(database_directory))
results.to_excel('shape_analysis_final.xlsx', index=False)
