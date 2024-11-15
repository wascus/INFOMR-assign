import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import os
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec

#load everything
file_path = "/Users/maggiemaliszewski/Desktop/ToGitHub/feature_vector.csv"
features_df = pd.read_csv(file_path)
obj_root_directory = "/Users/maggiemaliszewski/Desktop/ToGitHub/ShapeDatabase_processed"


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

feature_columns = single_value_features + [bin for group in histogram_features.values() for bin in group]
data_matrix = features_df[feature_columns].values

#train the k-NN model
top_k = 5
knn_model = KNeighborsClassifier(n_neighbors=top_k + 1, metric='euclidean')
knn_model.fit(data_matrix, features_df['Class'])


def render_model(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return np.zeros((100, 100, 3), dtype='uint8')  #keep blank if the image doesn't exist

    try:
        mesh = trimesh.load(file_path)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.add_collection3d(Poly3DCollection(mesh.triangles, alpha=0.7, facecolor='lightblue', edgecolor='k'))
        ax.auto_scale_xyz(mesh.bounds[:, 0], mesh.bounds[:, 1], mesh.bounds[:, 2])
        ax.axis('off')

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  #RGBA image with 4 channels
        plt.close(fig)
        return image[:, :, :3]  #remove alpha channel to convert to RGB
    except Exception as e:
        print(f"Error rendering {file_path}: {e}")
        return np.zeros((100, 100, 3), dtype='uint8')

#make the query and display everything in a plot
def search_and_display_multiple(query_shape_filenames):
    num_queries = len(query_shape_filenames)
    fig = plt.figure(figsize=(24, num_queries * 8))

    grid = gridspec.GridSpec(num_queries, 6, wspace=0, hspace=0) #ensure tight spacing for better lookign plot

    for query_idx, query_shape_filename in enumerate(query_shape_filenames):
        query_row = features_df[features_df['File'] == query_shape_filename]

        if query_row.empty:
            print(f"Shape '{query_shape_filename}' not found in the dataset.")
            continue

        query_vector = query_row[feature_columns].values
        distances, indices = knn_model.kneighbors(query_vector)
        top_matches = [(features_df.iloc[idx]['File'], features_df.iloc[idx]['Class'], dist) for idx, dist in zip(indices[0], distances[0])]

        #display query object
        query_model_path = os.path.join(obj_root_directory, query_row.iloc[0]['Class'], query_shape_filename)
        query_image = render_model(query_model_path)

        ax = fig.add_subplot(grid[query_idx, 0])
        ax.imshow(query_image)
        ax.axis('off')
        ax.set_title(f"{query_shape_filename}\nQuery Object", fontsize=16)

        #display top matches
        for col_idx, (model_file, model_class, model_distance) in enumerate(top_matches[1:6]):  #skip the first result (query object)
            model_path = os.path.join(obj_root_directory, model_class, model_file)
            model_image = render_model(model_path)

            ax = fig.add_subplot(grid[query_idx, col_idx + 1])
            ax.imshow(model_image)
            ax.axis('off')
            ax.set_title(f"{model_file}\nClass: {model_class}\nDist: {model_distance:.4f}", fontsize=12)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  #remove borders to make plot look better
    plt.show()

#specify queries:
query_shape_filenames = [
    "m1476.obj",
    "D01138.obj",
    "m1601.obj",
]
search_and_display_multiple(query_shape_filenames)
