import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance  # For EMD calculation

file_path = "feature_vector.csv"
master_folder = "ShapeDatabase_processed"
features_df = pd.read_csv(file_path)

#split the feature vector into single value and histogram features
single_value_features = [
    "Surface Area", "Volume", "Compactness", "Rectangularity",
    "Diameter", "Convexity", "Eccentricity"
]

histogram_features = {
    'A3': [f'A3_bin_{i}' for i in range(40)],
    'D1': [f'D1_bin_{i}' for i in range(40)],
    'D2': [f'D2_bin_{i}' for i in range(40)],
    'D3': [f'D3_bin_{i}' for i in range(40)],
    'D4': [f'D4_bin_{i}' for i in range(40)]
}

#define the feature weights for each histogram feature (single-value features are all set to 1.0)
feature_weights = {
    'A3': 1.0,
    'D1': 1000.0,
    'D2': 1000.0,
    'D3': 1000.0,
    'D4': 1000.0
}

# Define the distance function
def compute_distance(query, candidate, histogram_features, single_value_features, feature_weights):
    #emd for histogram features
    weighted_emd_sum = 0
    for feature, bins in histogram_features.items():
        query_hist = query[bins].values
        candidate_hist = candidate[bins].values
        emd = wasserstein_distance(query_hist, candidate_hist)
        weighted_emd_sum += feature_weights[feature] * emd

    #euclidean distance for single value features
    query_values = query[single_value_features].values
    candidate_values = candidate[single_value_features].values
    euclidean_dist = np.sqrt(np.sum((query_values - candidate_values) ** 2))

    return weighted_emd_sum + euclidean_dist


#find the closest matches
def query_shape(query_id, features_df, histogram_features, single_value_features, feature_weights, top_k=10):
    query = features_df.loc[features_df['File'] == query_id].iloc[0]
    distances = []
    for index, row in features_df.iterrows():
        if row['File'] != query_id:
            dist = compute_distance(query, row, histogram_features, single_value_features, feature_weights)
            distances.append((row['File'], row['Class'], dist))
    distances.sort(key=lambda x: x[2])
    return distances[:top_k]

#query shape and print results
query_id = 'm1473.obj'  #with file name
top_matches = query_shape(query_id, features_df, histogram_features, single_value_features, feature_weights)
print("Top matches:")
for file, cls, dist in top_matches:
    print(f"File: {file}, Class: {cls}, Distance: {dist:.4f}")