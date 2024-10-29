import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Load the normalized features from the provided CSV file
file_path = "merged_normalized_features_combined.csv"
features_df = pd.read_csv(file_path)

# Define the columns representing the features
feature_columns = [
    "Surface Area", "Volume", "Compactness", "Rectangularity", "Diameter",
    "Convexity", "Eccentricity", "A3", "D1", "D2", "D3", "D4"
]

# Input the query shape filename (replace 'D00309.obj' with your desired shape file)
query_shape_filename = 'D00265.obj'

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

    # Compute the Euclidean distance between the query vector and all database shapes
    distances = distance.cdist([query_vector], features_matrix, metric='euclidean')[0]

    # Add the distances to the DataFrame
    features_df['Distance'] = distances

    # Sort shapes by distance to find the closest matches
    top_k = 10  # Adjust this number as needed
    top_matches = features_df.sort_values(by='Distance').head(top_k)

    # Display the top matches
    print("Top matches for the query shape:")
    print(top_matches[['Class', 'File', 'Distance']])

    # Calculate the threshold for the top 10% of closest distances
    percentile_10_threshold = features_df['Distance'].quantile(0.10)

    # Filter shapes that have a distance below this threshold
    matches_within_top_10_percent = features_df[features_df['Distance'] <= percentile_10_threshold]

    print("\nShapes within the top 10% of closest distances:")
    print(matches_within_top_10_percent[['Class', 'File', 'Distance']])

    # Plot a histogram of the distances for visualization
    plt.figure(figsize=(8, 6))
    plt.hist(features_df['Distance'], bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Distances from Query Shape to Database Shapes')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
