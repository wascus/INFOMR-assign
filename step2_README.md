# OBJ Normalisation Script

This script normalises 3D models in OBJ format by translating, scaling, aligning, and resolving flipping issues.

## Features

- **Load OBJ**: Reads vertices, faces, and normals from an OBJ file.
- **Normalise**:
  - Translate to origin (barycentre).
  - Scale uniformly to a unit sphere.
  - Align to coordinate axes using covariance eigenvectors.
  - Resolve flipping based on geometric moments.
- **Export OBJ**: Writes the transformed model to an OBJ file.

## Usage

### Functions

- `get_baricenter(vertices)`: Computes barycentre.
- `translate_to_origin(vertices, baricenter)`: Translates to origin.
- `scale_uniformly(vertices)`: Scales within unit sphere.
- `alignment(vertices)`: Aligns to axes.
- `flipping(vertices)`: Fixes orientation.
- `normalize(vertices)`: Performs all steps above.
- `write_obj(filename, vertices, faces, ...)`: Exports OBJ file.

### Example

```python
from test import load_obj

vertices, faces, normals = load_obj('path/to/model.obj')
vertices = normalize(vertices)
write_obj('path/to/output.obj', vertices, faces)
