import open3d as o3d
import numpy as np


def get_baricenter(vertices):
    x = sum(v[0] for v in vertices) / len(vertices)
    y = sum(v[1] for v in vertices) / len(vertices)
    z = sum(v[2] for v in vertices) / len(vertices)
    return x, y, z


def translate_to_origin(vertices, baricenter):
    return [(v[0] - baricenter[0], v[1] - baricenter[1], v[2] - baricenter[2]) for v in vertices]


def scale_uniformly(vertices):
    max_dist = max([max(abs(v[0]), abs(v[1]), abs(v[2])) for v in vertices])
    return [(v[0] / max_dist, v[1] / max_dist, v[2] / max_dist) for v in vertices]


def alignment(vertices):
    # Compute the covariance matrix
    cov = np.cov(vertices, rowvar=False)
    # Compute the eigenvectors of the covariance matrix
    eigvals, eigvecs = np.linalg.eig(cov)
    # Compute the rotation matrix
    R = np.array([eigvecs[:, 0], eigvecs[:, 1], [0, 0, 1]])
    # Rotate the shape
    return [R @ v for v in vertices]


def flipping(vertices):
    # Compute the covariance matrix
    cov = np.cov(vertices, rowvar=False)
    # Compute the eigenvectors of the covariance matrix
    eigvals, eigvecs = np.linalg.eig(cov)
    # Compute the centroid
    centroid = np.mean(vertices, axis=0)
    # Compute the moments
    moments = np.zeros(3)
    for v in vertices:
        moments += np.square(v - centroid)
    moments /= len(vertices)
    # Compute the moments of the flipped shape
    flipped_moments = np.zeros(3)
    for v in vertices:
        flipped_moments += np.square(np.array([v[0], -v[1], -v[2]]) - centroid)
    flipped_moments /= len(vertices)
    # Compute the moment test
    moment_test = np.sum(np.square(moments - flipped_moments))
    # Flip the shape along the 3 axes
    if moment_test < 1e-6:
        return [[v[0], -v[1], -v[2]] for v in vertices]
    else:
        return vertices


def normalize_mesh(o3d_mesh):
    """
    Normalize the vertices of an Open3D mesh.

    :param o3d_mesh: Open3D mesh object to normalize.
    :return: Normalized Open3D mesh.
    """
    vertices = np.asarray(o3d_mesh.vertices)
    baricenter = get_baricenter(vertices)
    vertices = translate_to_origin(vertices, baricenter)
    vertices = scale_uniformly(vertices)
    vertices = alignment(vertices)
    vertices = flipping(vertices)

    # Update the Open3D mesh with normalized vertices
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return o3d_mesh


# Example function to integrate this into another script
def Normalize(o3d_mesh):
    """
    Normalize an Open3D mesh and return the processed mesh.

    :param o3d_mesh: Open3D mesh object to process.
    :return: Normalized Open3D mesh.
    """
    return normalize_mesh(o3d_mesh)
