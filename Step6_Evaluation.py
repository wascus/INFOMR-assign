import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance

# Define feature metadata and weights
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
feature_weights = {
    'A3': 1.0,
    'D1': 1000.0,
    'D2': 1000.0,
    'D3': 1000.0,
    'D4': 1000.0
}


def compute_distance(query, candidate, histogram_features, single_value_features, feature_weights):
    # EMD for histogram features
    weighted_emd_sum = 0
    for feature, bins in histogram_features.items():
        query_hist = query[bins].values
        candidate_hist = candidate[bins].values
        emd = wasserstein_distance(query_hist, candidate_hist)
        weighted_emd_sum += feature_weights[feature] * emd

    # Euclidean distance for single-value features
    query_values = query[single_value_features].values
    candidate_values = candidate[single_value_features].values
    euclidean_dist = np.sqrt(np.sum((query_values - candidate_values) ** 2))

    return weighted_emd_sum * euclidean_dist


def query_shape(query_features, database_df, histogram_features, single_value_features, feature_weights, top_k=10):
    distances = []
    for _, row in database_df.iterrows():
        dist = compute_distance(query_features, row, histogram_features, single_value_features, feature_weights)
        distances.append((row['File'], row['Class'], dist))
    distances.sort(key=lambda x: x[2])
    return distances[:top_k]


def evaluate_query(query_features, database_df, histogram_features, single_value_features, feature_weights, top_k=10):
    query_label = query_features['Class']
    top_matches = query_shape(query_features, database_df, histogram_features, single_value_features, feature_weights,
                              top_k)

    # Print the top matches for the query
    print(f"\nQuery: {query_features['File']}")
    print(f"Top {top_k} Matches:")
    for rank, (file, class_label, distance) in enumerate(top_matches, start=1):
        print(f"{rank}. {file} ({class_label}) - Distance: {distance:.4f}")

    # Count relevant and retrieved shapes
    retrieved_labels = [match[1] for match in top_matches]
    relevant_count = retrieved_labels.count(query_label)
    accuracy = relevant_count / top_k
    precision = relevant_count / len(database_df[database_df['Class'] == query_label])

    return precision, accuracy


def main():
    # Load the database CSV
    database_path = "feature_vector.csv"
    database_df = pd.read_csv(database_path)

    # Randomly select 100 models from the database for evaluation
    query_samples = database_df.sample(n=100, random_state=42)

    overall_precision = []
    overall_accuracy = []

    for _, query_features in query_samples.iterrows():
        # Evaluate query
        precision, accuracy = evaluate_query(query_features, database_df, histogram_features, single_value_features,
                                             feature_weights, top_k=10)
        overall_precision.append(precision)
        overall_accuracy.append(accuracy)

        print(f"Precision: {precision:.4f}, Accuracy: {accuracy:.4f}")

    # Compute overall metrics
    avg_precision = np.mean(overall_precision) if overall_precision else 0
    avg_accuracy = np.mean(overall_accuracy) if overall_accuracy else 0

    print("\nOverall Performance:")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")


if __name__ == "__main__":
    main()
