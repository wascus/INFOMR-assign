from test import load_obj
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
    # Align the shape so that the two largest eigenvectors match the x, respectively y, axes of the coordinate frame
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


def write_obj(filename, vertices, faces, normals=None, texcoords=None, group_name=None):
    with open(filename, 'w') as file:
        # Optional: write group name if provided
        if group_name:
            file.write(f"g {group_name}\n")
        
        # Write vertices (v)
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Write texture coordinates (vt), if provided
        if texcoords is not None:
            for texcoord in texcoords:
                file.write(f"vt {texcoord[0]} {texcoord[1]}\n")
        
        # Write vertex normals (vn), if provided
        if normals is not None:
            for normal in normals:
                file.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
        
        # Write faces (f) with optional texture and normal indices
        for face in faces:
            face_str = []
            for vertex_idx in face:
                if isinstance(vertex_idx, int):  # If face contains just vertex indices
                    face_str.append(f"{vertex_idx + 1}")  # 1-indexed for OBJ format
                else:
                    # Handle vertex/texture/normal triplets
                    vertex_part = str(vertex_idx[0] + 1)  # OBJ files are 1-indexed
                    texcoord_part = str(vertex_idx[1] + 1) if vertex_idx[1] is not None else ''
                    normal_part = str(vertex_idx[2] + 1) if len(vertex_idx) > 2 and vertex_idx[2] is not None else ''
                    
                    # Combine the indices into "v/t/n" format
                    if texcoord_part or normal_part:
                        face_str.append(f"{vertex_part}/{texcoord_part}/{normal_part}")
                    else:
                        face_str.append(f"{vertex_part}")
            file.write(f"f {' '.join(face_str)}\n")


def normalize(vertices):
    baricenter = get_baricenter(vertices)
    vertices = translate_to_origin(vertices, baricenter)
    vertices = scale_uniformly(vertices)
    vertices = alignment(vertices)
    vertices = flipping(vertices)
    return vertices

if __name__ == '__main__':
    vertices, faces, normals = load_obj('ShapeDatabase_INFOMR/Bus/D00769.obj')
    vertices = normalize(vertices)
    write_obj('output2.obj', vertices, faces)
