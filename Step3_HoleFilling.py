import os
import gc
import trimesh
import numpy as np
from collections import defaultdict
import pymeshlab

#update as necessary
database_path = '/Users/maggiemaliszewski/INFOMR-assign/ShapeDatabase_INFOMR-master-normalized'
fixed_dataset_path = 'ShapeDatabase_INFOMR-master-finalattempt'

#create a folder for the fixed dataset
os.makedirs(fixed_dataset_path, exist_ok=True)

#initialize counters that are used to evaluate the hole filling algorithms
no_holes_count = 0
initial_holes_count = 0
filled_by_first_method_count = 0
fixed_by_trimesh_count = 0
fixed_by_pymeshlab_count = 0
still_with_holes_count = 0

#find boundary edges that appear in only one triangle, to identify holes
def find_boundary_edges(mesh):
    edge_count = defaultdict(int)

    for face in mesh.faces:
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        for edge in edges:
            edge = tuple(sorted(edge))  # Store edges in a sorted order
            edge_count[edge] += 1

    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    return boundary_edges

def clean_mesh(mesh):
    edge_count = defaultdict(int)

    for face in mesh.faces:
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        for edge in edges:
            edge = tuple(sorted(edge))  # Store edges in a sorted order
            edge_count[edge] += 1

    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    return mesh

#organize boundary edges into hole boundaries
def find_hole_boundaries(boundary_edges):
    edge_map = defaultdict(list)
    for edge in boundary_edges:
        v1, v2 = edge
        edge_map[v1].append(v2)
        edge_map[v2].append(v1)

    hole_boundaries = []
    visited_edges = set()

    for edge in boundary_edges:
        if edge not in visited_edges:
            boundary = []
            v1, v2 = edge
            current_edge = edge
            start_vertex = v1
            current_vertex = v2

            boundary.append(current_edge)
            visited_edges.add(current_edge)

            while current_vertex != start_vertex: #traverse the loop starting with this edge
                next_vertex = None
                for neighbor in edge_map[current_vertex]:
                    potential_edge = tuple(sorted((current_vertex, neighbor)))
                    if potential_edge not in visited_edges:
                        next_vertex = neighbor
                        current_edge = potential_edge
                        break

                if next_vertex:
                    boundary.append(current_edge)
                    visited_edges.add(current_edge)
                    current_vertex = next_vertex
                else:
                    break

            hole_boundaries.append(boundary)

    return hole_boundaries

#use the method from Technical Tips Step3 to fill the holes (https://webspace.science.uu.nl/~telea001/uploads/MR/03b-TechTips.pdf)
def fill_holes(mesh, hole_boundaries):

    for boundary in hole_boundaries:
        vertices = list({v for edge in boundary for v in edge})

        #compute barrycenter of hole
        hole_vertices = np.array([mesh.vertices[v] for v in vertices])
        barycenter = np.mean(hole_vertices, axis=0)

        #add barycenter as a new vertex to the mesh
        barycenter_index = len(mesh.vertices)
        mesh.vertices = np.vstack([mesh.vertices, barycenter])

        #create a triangle fan to fill the hole
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]

            #ensure consistent normal direction by ordering vertices counterclockwise
            normal_check = np.cross(mesh.vertices[v2] - mesh.vertices[v1], barycenter - mesh.vertices[v1])
            normal_direction = np.dot(mesh.face_normals.mean(axis=0), normal_check)

            if normal_direction > 0:
                new_face = [v1, v2, barycenter_index]  # Correct order
            else:
                new_face = [v2, v1, barycenter_index]  # Reverse order

            mesh.faces = np.vstack([mesh.faces, new_face])

def pymeshlabrepair(ms):
    # Remove non-manifold edges
    ms.apply_filter('meshing_repair_non_manifold_edges')
    print("Removed non-manifold edges.")
    ms.apply_filter('meshing_close_holes', maxholesize=1000)
    return ms.current_mesh()  # Return the modified mesh
    return ms

#use boundary edges to check if mesh has holes (watertight = no holes)
def is_mesh_watertight(mesh):
    boundary_edges = find_boundary_edges(mesh) #no boundary edges = mesh is watertight
    return len(boundary_edges) == 0

def process_mesh(mesh_file, class_folder):

    global no_holes_count, initial_holes_count, filled_by_first_method_count, fixed_by_trimesh_count, fixed_by_pymeshlab_count, still_with_holes_count

    #create class folders in the new directory
    class_fixed_path = os.path.join(fixed_dataset_path, class_folder)
    os.makedirs(class_fixed_path, exist_ok=True)

    fixed_mesh_path = os.path.join(class_fixed_path, os.path.basename(mesh_file))

    #skip this file if it was already processed
    #necessary in case code has to be rerun, eg due to a crash, to ensure no file duplicates
    if os.path.exists(fixed_mesh_path):
        print(f"{fixed_mesh_path} already exists. Skipping.")
        return

    #open mesh using pymeshlab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_file)

    #clean up mesh
    ms.apply_filter('meshing_remove_duplicate_vertices')
    ms.apply_filter('meshing_remove_duplicate_faces')

    #translate mesh into trimesh (needed for watertightness check, and two hole filling methods)
    vertices = ms.current_mesh().vertex_matrix()
    triangles = ms.current_mesh().face_matrix()
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)


    #find holes, if mesh is watertight - copy it directly to the new directory
    boundary_edges = find_boundary_edges(trimesh_mesh)
    if len(boundary_edges) == 0:
        no_holes_count += 1
        trimesh_mesh.export(fixed_mesh_path)
        return


    #if we find holes, locate hole boundaries
    initial_holes_count += 1
    hole_boundaries = find_hole_boundaries(boundary_edges)


    #attempt to fill holes: first, use the method from technical tips
    fill_holes(trimesh_mesh, hole_boundaries)


    #check if this worked
    if is_mesh_watertight(trimesh_mesh):
        print("Mesh is now watertight.")
        filled_by_first_method_count += 1
        trimesh_mesh.export(fixed_mesh_path)
        return
    else:
        #if holes are still present: attempt to fill using trimesh.repair.fill_holes()
        trimesh.repair.fill_holes(trimesh_mesh)

        #check if worked
        if is_mesh_watertight(trimesh_mesh):
            print("Mesh is watertight after trimesh repair.")
            fixed_by_trimesh_count += 1
        else:
            print("Mesh still has holes after pymeshlab attempt.")

        #try to fill holes using pymeshlab
        pymeshlabrepair(ms)

        #translate the new mesh into trimesh to check watertightness
        vertices = ms.current_mesh().vertex_matrix()
        triangles = ms.current_mesh().face_matrix()
        new_trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

        # check if worked
        if is_mesh_watertight(new_trimesh_mesh):
            print("Mesh is watertight after pymeshlab repair.")
            fixed_by_pymeshlab_count += 1
        else:
            print("Mesh still has holes after repair attempt.")
        still_with_holes_count += 1

        #move to new directory
        new_trimesh_mesh.export(fixed_mesh_path)

#process the database
def process_all_meshes():
    for class_folder in os.listdir(database_path):
        class_folder_path = os.path.join(database_path, class_folder)
        if os.path.isdir(class_folder_path):
            for root, _, files in os.walk(class_folder_path):
                for file in files:
                    if file.endswith('.obj'):
                        mesh_file = os.path.join(root, file)
                        process_mesh(mesh_file, class_folder)
                        gc.collect()

    #print data about how well hole filling went
    print("\nProcessing complete.")
    print(f"Meshes with no holes: {no_holes_count}")
    print(f"Meshes with holes initially: {initial_holes_count}")
    print(f"Meshes fixed by first method: {filled_by_first_method_count}")
    print(f"Meshes fixed by trimesh: {fixed_by_trimesh_count}")
    print(f"Meshes fixed by pymeshlab: {fixed_by_pymeshlab_count}")
    print(f"Meshes still with holes after repair attempts: {still_with_holes_count}")

if __name__ == "__main__":
    process_all_meshes()
