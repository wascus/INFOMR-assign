import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from pyemd import emd
import os

#load all the files
file_path = "feature_vector.csv"
master_folder = "ShapeDatabase_processed"
features_df = pd.read_csv(file_path)

#divide into single value and histogram features
single_value_features = [
    "Surface Area", "Volume", "Compactness", "Rectangularity", "Diameter",
    "Convexity", "Eccentricity"
]
histogram_features = {
    'A3': [f'A3_bin_{i}' for i in range(40)],
    'D1': [f'D1_bin_{i}' for i in range(40)],
    'D2': [f'D2_bin_{i}' for i in range(40)],
    'D3': [f'D3_bin_{i}' for i in range(40)],
    'D4': [f'D4_bin_{i}' for i in range(40)]
}

#extract histogram features for each row
def get_histogram_vector(row):
    return np.concatenate([row[feature_bins].values.flatten() for feature_bins in histogram_features.values()])

#precompute distance ranges per feature for distance normalization
def calculate_distance_ranges(features_matrix):
    ranges = []
    for i in range(features_matrix.shape[1]):
        feature_column = features_matrix[:, i]
        distances = np.abs(feature_column[:, None] - feature_column)
        range_val = np.max(distances) - np.min(distances)
        ranges.append(range_val if range_val != 0 else 1)
    return np.array(ranges)

#calculate euclidean distance
def calculate_l2_distance(query_vector, features_matrix):
    return np.sqrt(np.sum((features_matrix - query_vector) ** 2, axis=1))

#calculate EMD with distance weighting for histogram features
def calculate_emd(query_vector, features_matrix, distance_ranges):
    distance_matrix = np.ones((len(query_vector), len(query_vector))) - np.eye(len(query_vector))
    emd_distances = []
    for feature_vector in features_matrix:
        normalized_query = query_vector / distance_ranges
        normalized_feature_vector = feature_vector / distance_ranges
        emd_distances.append(
            emd(normalized_query.astype(np.float64), normalized_feature_vector.astype(np.float64), distance_matrix)
        )
    return emd_distances

#render the model
def render_model(file_path):
    mesh = trimesh.load(file_path)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    #create a wireframe plot of the mesh
    vertices = mesh.vertices
    faces = mesh.faces
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2], color='lightblue', edgecolor='k',
                    linewidth=0.3)
    ax.axis('off')

    #convert from matplotlib rendering into an image using buffer_rgba
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA image with 4 channels
    plt.close(fig)

    return image

#search and then make and show the visualization:
def search_and_display_multiple(query_shape_filenames):

    fig, axes = plt.subplots(len(query_shape_filenames), 7, figsize=(20, len(query_shape_filenames) * 6))

    for row_idx, query_shape_filename in enumerate(query_shape_filenames):
        query_row = features_df[features_df['File'] == query_shape_filename]

        if query_row.empty:
            print(f"Shape '{query_shape_filename}' not found in the dataset.")
            continue
        else:
            query_vector_single = query_row[single_value_features].values.flatten()
            query_vector_histogram = get_histogram_vector(query_row)

            features_matrix_single = features_df[single_value_features].values
            features_matrix_histogram = np.array([get_histogram_vector(row) for _, row in features_df.iterrows()])

            #calculate the euclidean and emd distances
            l2_distances = calculate_l2_distance(query_vector_single, features_matrix_single)
            distance_ranges = calculate_distance_ranges(features_matrix_histogram)
            emd_distances = calculate_emd(query_vector_histogram, features_matrix_histogram, distance_ranges)

            #get the top 6 matches
            features_df['Distance'] = l2_distances + emd_distances
            top_matches = features_df.sort_values(by='Distance').head(6)

            #first show the query object seperately
            query_model_path = os.path.join(master_folder, query_row.iloc[0]['Class'], query_shape_filename)
            query_image = render_model(query_model_path)
            axes[row_idx, 0].imshow(query_image)
            axes[row_idx, 0].axis('off')
            axes[row_idx, 0].set_title(f"{query_shape_filename}\nQuery Object", fontsize=12)

            for col_idx, (_, row) in enumerate(top_matches.iterrows()):
                if col_idx >= 6:
                    break  #show the 5 closest matches

                model_file = row['File']
                model_class = row['Class']
                model_distance = row['Distance']
                model_path = os.path.join(master_folder, model_class, model_file)

                #display everything
                model_image = render_model(model_path)
                axes[row_idx, col_idx + 1].imshow(model_image)
                axes[row_idx, col_idx + 1].axis('off')
                axes[row_idx, col_idx + 1].set_title(f"{model_file}\nClass: {model_class}\nDist: {model_distance:.4f}", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#specify queries:
query_shape_filenames = [
    "m1476.obj",
    "D01138.obj",
    "m1601.obj",
]
search_and_display_multiple(query_shape_filenames)
