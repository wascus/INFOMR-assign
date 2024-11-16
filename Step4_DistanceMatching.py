import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from pyemd import emd
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec

#load everything
file_path = "feature_vector.csv"
features_df = pd.read_csv(file_path)
obj_root_directory = "ShapeDatabase_INFOMR-master"


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


def render_model(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return np.zeros((100, 100, 3), dtype='uint8')   #keep blank if the image doesn't exist

    try:
        mesh = trimesh.load(file_path)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.add_collection3d(Poly3DCollection(mesh.triangles, alpha=0.7, facecolor='lightblue', edgecolor='k'))
        ax.auto_scale_xyz(mesh.bounds[:, 0], mesh.bounds[:, 1], mesh.bounds[:, 2])
        ax.axis('off')

        # Use buffer_rgba() to get the image data
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  #RGBA image with 4 channels
        plt.close(fig)
        return image[:, :, :3]  #remove alpha channel to convert to RGB
    except Exception as e:
        print(f"Error rendering {file_path}: {e}")
        return np.zeros((100, 100, 3), dtype='uint8')

def search_and_display_multiple(query_shape_filenames):
    num_queries = len(query_shape_filenames)
    fig = plt.figure(figsize=(24, num_queries * 8))

    grid = gridspec.GridSpec(num_queries, 6, wspace=0, hspace=0) #ensure tight spacing for better lookign plot

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

            #calculate the distances
            l2_distances = calculate_l2_distance(query_vector_single, features_matrix_single)
            distance_ranges = calculate_distance_ranges(features_matrix_histogram)
            emd_distances = calculate_emd(query_vector_histogram, features_matrix_histogram, distance_ranges)

            #display top matches
            features_df['Distance'] = l2_distances + emd_distances
            top_matches = features_df.sort_values(by='Distance').iloc[1:6]  #skip the first result (query object)

            query_model_path = os.path.join(obj_root_directory, query_row.iloc[0]['Class'], query_shape_filename)
            query_image = render_model(query_model_path)

            ax = fig.add_subplot(grid[row_idx, 0])
            ax.imshow(query_image)
            ax.axis('off')
            ax.set_title(f"{query_shape_filename}\nQuery Object", fontsize=16)

            for col_idx, (_, row) in enumerate(top_matches.iterrows()):
                model_file = row['File']
                model_class = row['Class']
                model_distance = row['Distance']
                model_path = os.path.join(obj_root_directory, model_class, model_file)

                model_image = render_model(model_path)
                ax = fig.add_subplot(grid[row_idx, col_idx + 1])
                ax.imshow(model_image)
                ax.axis('off')
                ax.set_title(f"{model_file}\nClass: {model_class}\nDist: {model_distance:.4f}", fontsize=12)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  #remove borders to make plot look better
    plt.show()


#specify queries:
query_shape_filenames = [
    "D00078.obj"
]
search_and_display_multiple(query_shape_filenames)
