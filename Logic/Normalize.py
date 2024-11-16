import numpy as np
import open3d as o3d

def get_baricenter(vertices):
    """
    Compute the barycenter (centroid) of the vertices.
    """
    x = sum(v[0] for v in vertices) / len(vertices)
    y = sum(v[1] for v in vertices) / len(vertices)
    z = sum(v[2] for v in vertices) / len(vertices)
    return x, y, z


def translate_to_origin(vertices, baricenter):
    """
    Translate vertices so that the barycenter is at the origin.
    """
    return [(v[0] - baricenter[0], v[1] - baricenter[1], v[2] - baricenter[2]) for v in vertices]


def scale_uniformly(vertices):
    """
    Scale vertices uniformly so that the maximum distance from the origin is 1.
    """
    max_dist = max([max(abs(v[0]), abs(v[1]), abs(v[2])) for v in vertices])
    return [(v[0] / max_dist, v[1] / max_dist, v[2] / max_dist) for v in vertices]


def apply_precision(vertices, epsilon=1e-6):
    """
    Round vertex coordinates to avoid small numerical errors.
    """
    return [(round(v[0], 6), round(v[1], 6), round(v[2], 6)) for v in vertices]


def alignment(vertices):
    """
    Align the mesh along its principal axes using covariance matrix eigenvectors.
    Ensures a right-handed system.
    """
    # Compute covariance matrix
    cov = np.cov(np.array(vertices).T)  # Transpose for correct shape

    # Compute eigenvectors and eigenvalues
    eigvals, eigvecs = np.linalg.eig(cov)

    # Sort eigenvectors by eigenvalue magnitude
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sorted_indices]

    # Ensure right-handed system
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] = -eigvecs[:, -1]

    # Construct the rotation matrix
    R = np.array([eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]])

    # Rotate vertices
    vertices = [R @ np.array(v) for v in vertices]

    # Apply precision rounding
    vertices = apply_precision(vertices)

    return vertices


def flipping(vertices):
    """
    Flip the mesh along certain axes to ensure symmetry.
    """
    # Compute covariance matrix
    cov = np.cov(vertices, rowvar=False)

    # Compute eigenvectors
    eigvals, eigvecs = np.linalg.eig(cov)

    # Compute centroid
    centroid = np.mean(vertices, axis=0)

    # Compute moments
    moments = np.zeros(3)
    for v in vertices:
        moments += np.square(v - centroid)
    moments /= len(vertices)

    # Compute moments of flipped shape
    flipped_moments = np.zeros(3)
    for v in vertices:
        flipped_moments += np.square(np.array([v[0], -v[1], -v[2]]) - centroid)
    flipped_moments /= len(vertices)

    # Test for symmetry
    moment_test = np.sum(np.square(moments - flipped_moments))

    if moment_test < 1e-6:
        return [[v[0], -v[1], -v[2]] for v in vertices]
    else:
        return vertices


def Normalize(mesh):
    """
    Normalize an Open3D mesh by translating to origin, scaling uniformly, aligning to principal axes, and flipping.
    :param mesh: Open3D mesh object
    :return: Normalized Open3D mesh
    """
    vertices = np.asarray(mesh.vertices)

    # Compute barycenter and translate to origin
    baricenter = get_baricenter(vertices)
    vertices = translate_to_origin(vertices, baricenter)

    # Scale uniformly
    vertices = scale_uniformly(vertices)

    # Align to principal axes
    vertices = alignment(vertices)

    # Ensure symmetry with flipping
    vertices = flipping(vertices)

    # Update the mesh with normalized vertices
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    return mesh
