import open3d as o3d
import numpy as np
import pymeshlab
import tempfile


def compute_surface_integral_volume(mesh):
    """
    Compute the volume of a mesh using surface integrals, even for non-watertight meshes.
    """
    volume = 0.0
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    for tri in triangles:
        v1 = vertices[tri[0]]
        v2 = vertices[tri[1]]
        v3 = vertices[tri[2]]

        normal = np.cross(v2 - v1, v3 - v1)  # Calculate the normal of the triangle
        volume += np.dot(normal, v1)  # Calculate the signed volume of the tetrahedron

    return abs(volume) / 6.0  # Divide by 6 based on tetrahedron volume formula


def compute_convex_hull_volume(mesh):
    """
    Compute the convex hull volume using PyMeshLab for non-watertight meshes.
    """
    try:
        # Convert the Open3D mesh to vertices and triangles
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # Create a temporary directory to save and load the mesh
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_mesh_path = f"{temp_dir}/temp_mesh.obj"

            # Write the mesh to a temporary OBJ file
            with open(temp_mesh_path, "w") as f:
                # Write vertices
                for v in vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")

                # Write faces (OBJ indices are 1-based)
                for tri in triangles:
                    f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")

            # Initialize a PyMeshLab MeshSet and load the mesh
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(temp_mesh_path)

            # Apply the convex hull filter
            ms.apply_filter('generate_convex_hull')

            # Get the convex hull mesh
            convex_hull_mesh = ms.current_mesh()

            # Calculate volume of the convex hull
            volume = 0.0
            vertices = convex_hull_mesh.vertex_matrix()
            faces = convex_hull_mesh.face_matrix()

            for face in faces:
                v1 = vertices[face[0]]
                v2 = vertices[face[1]]
                v3 = vertices[face[2]]

                normal = np.cross(v2 - v1, v3 - v1)
                volume += np.dot(normal, v1)

            return abs(volume) / 6.0
    except Exception as e:
        print(f"Error computing convex hull volume: {e}")
        return None


def GlobalDescriptors(mesh):
    """
    Compute global property descriptors for a given Open3D mesh.
    :param mesh: Open3D mesh object.
    :return: Dictionary containing descriptors.
    """
    descriptors = {}

    # Ensure consistent orientation of triangles and compute normals
    mesh.orient_triangles()
    mesh.compute_vertex_normals()

    # Surface Area
    descriptors['Surface Area'] = mesh.get_surface_area()

    # Volume using surface integrals
    volume = compute_surface_integral_volume(mesh)
    descriptors['Volume'] = volume

    # Compactness
    if descriptors['Surface Area'] > 0:
        descriptors['Compactness'] = (36 * np.pi * (volume ** 2)) / (descriptors['Surface Area'] ** 3)
    else:
        descriptors['Compactness'] = None

    # Rectangularity (calculated but not stored as Bounding Box Volume)
    bounding_box = mesh.get_oriented_bounding_box()
    bounding_box_volume = bounding_box.volume()
    descriptors['Rectangularity'] = volume / bounding_box_volume if bounding_box_volume > 0 else None

    # Diameter
    extents = bounding_box.extent
    descriptors['Diameter'] = np.max(extents)

    # Convexity
    convex_hull_volume = compute_convex_hull_volume(mesh)
    if convex_hull_volume is not None and convex_hull_volume > 0:
        descriptors['Convexity'] = volume / convex_hull_volume
    else:
        descriptors['Convexity'] = None

    # Eccentricity
    points = np.asarray(mesh.vertices)
    cov_matrix = np.cov(points.T)
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    if eigenvalues.min() > 0:
        descriptors['Eccentricity'] = eigenvalues.max() / eigenvalues.min()
    else:
        descriptors['Eccentricity'] = None

    return descriptors
