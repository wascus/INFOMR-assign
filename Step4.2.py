import pandas as pd
import matplotlib.pyplot as plt
from pyemd import emd
import numpy as np

# Load the normalized features from the provided CSV file
file_path = "merged_normalized_features_combined.csv"
features_df = pd.read_csv(file_path)

# Define the columns representing the features
feature_columns = [
    "Surface Area", "Volume", "Compactness", "Rectangularity", "Diameter",
    "Convexity", "Eccentricity", "A3", "D1", "D2", "D3", "D4"
]

# Input the query shape filename (replace 'D00309.obj' with your desired shape file)
query_shape_filename = 'D00397.obj'

# Extract the feature vector for the given query shape from the DataFrame
query_row = features_df[features_df['File'] == query_shape_filename]

# Check if the query shape exists in the CSV
if query_row.empty:
    print(f"Shape '{query_shape_filename}' not found in the dataset.")
else:
    # Convert the row to a list of feature values
    query_vector = query_row[feature_columns].values.flatten()

    # Drop the non-feature columns to get the features matrix
    features_matrix = features_df[feature_columns].values

    # Prepare the distance matrix for EMD calculation (since it's symmetric, we can use a simple one)
    # For simplicity, we'll use the identity matrix as the distance between each feature.
    distance_matrix = np.ones((len(query_vector), len(query_vector))) - np.eye(len(query_vector))

    # Compute the EMD between the query vector and all other shapes
    emd_distances = []
    for feature_vector in features_matrix:
        emd_distances.append(emd(query_vector.astype(np.float64), feature_vector.astype(np.float64), distance_matrix))

    # Add the EMD distances to the DataFrame
    features_df['Distance'] = emd_distances

    # Sort shapes by distance to find the closest matches
    top_k = 20  # Adjust this number as needed
    top_matches = features_df.sort_values(by='Distance').head(top_k)

    # Display the top matches
    print("Top matches for the query shape:")
    print(top_matches[['Class', 'File', 'Distance']])

    # Plot a histogram of the distances for visualization
    plt.figure(figsize=(8, 6))
    plt.hist(features_df['Distance'], bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of EMD Distances from Query Shape to Database Shapes')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
