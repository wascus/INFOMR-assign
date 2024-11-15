import pandas as pd
import numpy as np
from pyemd import emd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import pywavefront
import os

# Load normalized feature data
file_path = "feature_vector.csv"
features_df = pd.read_csv(file_path)

# Define single-value and histogram feature groups
single_value_features = ["Surface Area", "Volume", "Compactness", "Rectangularity", "Diameter", "Convexity",
                         "Eccentricity"]
histogram_groups = {
    'A3': [f'A3_bin_{i}.0' for i in range(10)],
    'D1': [f'D1_bin_{i}.0' for i in range(10)],
    'D2': [f'D2_bin_{i}.0' for i in range(10)],
    'D3': [f'D3_bin_{i}.0' for i in range(10)],
    'D4': [f'D4_bin_{i}.0' for i in range(10)]
}

# List of query shapes to visualize
#query_shape_filenames = ['D00265.obj', 'D00309.obj', 'D00123.obj', 'D00456.obj', 'D00567.obj']
#query_shape_filenames = ['D00956.obj', 'm736.obj', 'm252.obj', 'm1237.obj', 'D00850.obj']
#BAD - 'm1601.obj
query_shape_filenames = ['m359.obj', 'm280.obj', 'm1604.obj', 'm1609.obj', 'D01123.obj']
top_k = 5
obj_root_directory = "ShapeDatabase_INFOMR-master-finalattempt"  # Root directory with class-based subdirectories


# Helper function to compute L2 distance for single-value features
def calculate_l2(row_a, row_b, features):
    return np.linalg.norm(row_a[features].values - row_b[features].values)


# Helper function to compute EMD for histogram groups
def calculate_emd(query_vector, features_matrix, distance_ranges):
    distance_matrix = np.ones((len(query_vector), len(query_vector))) - np.eye(len(query_vector))
    emd_distances = []
    for feature_vector in features_matrix:
        # Apply distance weighting by dividing each feature by its precomputed range
        normalized_query = query_vector / distance_ranges
        normalized_feature_vector = feature_vector / distance_ranges
        emd_distances.append(
            emd(normalized_query.astype(np.float64), normalized_feature_vector.astype(np.float64), distance_matrix)
        )
    return emd_distances

def cosine_distance(row_a, row_b, features):
    return cosine(row_a[features].values, row_b[features].values)

# Combined function for calculating weighted distance
def combined_distance(row_a, row_b, single_value_features, histogram_groups):
    total_distance = 0.0

    # L2 distance for single-value features
    l2_dist = calculate_l2(row_a, row_b, single_value_features)
    total_distance += l2_dist

    # EMD for histogram groups
    for group_name, feature_bins in histogram_groups.items():
        emd_dist = calculate_emd(row_a, row_b, feature_bins)
        total_distance += emd_dist

    return total_distance


# Create figure for visualization
fig = plt.figure(figsize=(20, len(query_shape_filenames) * 5))
fig.suptitle('Query Shapes and Closest Matches: EMD and L2', fontsize=16)

# Process each query shape
for query_idx, query_shape_filename in enumerate(query_shape_filenames):
    query_row = features_df[features_df['File'] == query_shape_filename]

    if query_row.empty:
        print(f"Shape '{query_shape_filename}' not found in the dataset.")
        continue

    # Calculate distances from the query shape to all others
    distances = []
    for i, row in features_df.iterrows():
        dist = combined_distance(query_row.iloc[0], row, single_value_features, histogram_groups)
        distances.append((row['File'], row['Class'], dist))

    # Sort by distance and retrieve top matches
    top_matches = sorted(distances, key=lambda x: x[2])[:top_k + 1]

    # Print and plot top matches
    print(f"\nTop matches for query shape '{query_shape_filename}':")
    for match in top_matches:
        print(f"Class: {match[1]}, File: {match[0]}, Distance: {match[2]:.2f}")

    for i, (shape_filename, class_name, distance) in enumerate(top_matches):
        obj_path = os.path.join(obj_root_directory, class_name, shape_filename)

        subplot_idx = query_idx * (top_k + 1) + i + 1
        ax = fig.add_subplot(len(query_shape_filenames), top_k + 1, subplot_idx, projection='3d')

        title = "Query Shape" if i == 0 else f"Match {i}"
        ax.set_title(f"{title}\nClass: {class_name}\nDist: {distance:.2f}")

        if os.path.exists(obj_path):
            scene = pywavefront.Wavefront(obj_path, create_materials=True, collect_faces=True)
            vertices = np.array(scene.vertices)
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        else:
            ax.text(0.5, 0.5, "File\nNot Found", ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
