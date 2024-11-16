import trimesh
import numpy as np
import pymeshlab
import open3d as o3d
from collections import defaultdict


def find_boundary_edges(mesh):
    """
    Find boundary edges in a trimesh mesh.
    """
    edge_count = defaultdict(int)
    for face in mesh.faces:
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        for edge in edges:
            edge = tuple(sorted(edge))  # Store edges in a sorted order
            edge_count[edge] += 1
    return [edge for edge, count in edge_count.items() if count == 1]


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

            while current_vertex != start_vertex:  # Traverse the loop starting with this edge
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


def fill_holes(mesh, hole_boundaries):
    """
    Fill holes in the mesh using a triangle fan approach.
    """
    for boundary in hole_boundaries:
        vertices = list({v for edge in boundary for v in edge})

        # Compute barycenter of the hole
        hole_vertices = np.array([mesh.vertices[v] for v in vertices])
        barycenter = np.mean(hole_vertices, axis=0)

        # Add the barycenter as a new vertex to the mesh
        barycenter_index = len(mesh.vertices)
        mesh.vertices = np.vstack([mesh.vertices, barycenter])

        # Create a triangle fan to fill the hole
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]

            # Ensure consistent normal direction by ordering vertices counterclockwise
            normal_check = np.cross(mesh.vertices[v2] - mesh.vertices[v1], barycenter - mesh.vertices[v1])
            normal_direction = np.dot(mesh.face_normals.mean(axis=0), normal_check)

            if normal_direction > 0:
                new_face = [v1, v2, barycenter_index]
            else:
                new_face = [v2, v1, barycenter_index]

            mesh.faces = np.vstack([mesh.faces, new_face])


def is_mesh_watertight(mesh):
    """
    Check if a trimesh mesh is watertight by finding boundary edges.
    """
    boundary_edges = find_boundary_edges(mesh)
    return len(boundary_edges) == 0


def pymeshlab_repair(mesh):
    """
    Repair the mesh using PyMeshLab's filters.
    """
    try:
        ms = pymeshlab.MeshSet()
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))

        # Apply filters
        ms.apply_filter('meshing_repair_non_manifold_edges')
        ms.apply_filter('meshing_close_holes', maxholesize=1000)

        # Extract the repaired mesh
        repaired_mesh = ms.current_mesh()
        repaired_vertices = repaired_mesh.vertex_matrix()
        repaired_faces = repaired_mesh.face_matrix()

        return trimesh.Trimesh(vertices=repaired_vertices, faces=repaired_faces)
    except Exception as e:
        print(f"Error repairing mesh with PyMeshLab: {e}")
        return mesh


def HoleFilling(mesh):
    """
    Attempt to fill holes in a single Open3D mesh and return the repaired mesh.

    :param mesh: Open3D mesh object to process.
    :return: Repaired Open3D mesh.
    """
    # Convert Open3D mesh to Trimesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Check if the mesh is already watertight
    if is_mesh_watertight(trimesh_mesh):
        print("Mesh is already watertight.")
        return mesh

    # Find boundary edges and hole boundaries
    boundary_edges = find_boundary_edges(trimesh_mesh)
    hole_boundaries = find_hole_boundaries(boundary_edges)

    # Attempt to fill holes using the triangle fan approach
    fill_holes(trimesh_mesh, hole_boundaries)

    # Check if the mesh is now watertight
    if is_mesh_watertight(trimesh_mesh):
        print("Mesh is watertight after triangle fan hole filling.")
    else:
        # If not watertight, attempt to repair using trimesh's built-in method
        trimesh.repair.fill_holes(trimesh_mesh)

        if is_mesh_watertight(trimesh_mesh):
            print("Mesh is watertight after trimesh repair.")
        else:
            # If still not watertight, use PyMeshLab
            trimesh_mesh = pymeshlab_repair(trimesh_mesh)

            if is_mesh_watertight(trimesh_mesh):
                print("Mesh is watertight after PyMeshLab repair.")
            else:
                print("Mesh still has holes after all repair attempts.")

    # Convert back to Open3D mesh and return
    repaired_mesh = o3d.geometry.TriangleMesh()
    repaired_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    repaired_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    return repaired_mesh
