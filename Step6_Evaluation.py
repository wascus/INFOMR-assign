import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def evaluate_search_methods(file_path, knn_model, features_df, feature_columns, top_k=5):
    """
    Evaluate search_knn and search_emd methods using a file of features, measuring accuracy and precision.
    """
    # Load the input features
    input_features = pd.read_csv(file_path)

    knn_correct_predictions = 0
    emd_correct_predictions = 0
    knn_precisions = []
    emd_precisions = []

    total_queries = len(input_features)

    for _, query_row in input_features.iterrows():
        query_vector = query_row[feature_columns].values.reshape(1, -1)
        query_class = query_row['Class']
        query_file = query_row['File']

        # k-NN Search
        distances, indices = knn_model.kneighbors(query_vector, n_neighbors=top_k + 1)
        retrieved_classes_knn = features_df.iloc[indices[0][1:top_k + 1]]['Class']

        # Accuracy: Check if the majority class matches the query class
        if retrieved_classes_knn.mode()[0] == query_class:
            knn_correct_predictions += 1

        # Precision: Proportion of correct predictions in the top 5
        true_positives_knn = sum(retrieved_classes_knn == query_class)
        precision_knn = true_positives_knn / top_k
        knn_precisions.append(precision_knn)

        # EMD Search
        try:
            emd_results = search_with_emd(query_file)
            retrieved_classes_emd = [features_df.iloc[idx]['Class'] for idx in emd_results[:top_k]]

            # Accuracy: Check if the majority class matches the query class
            if pd.Series(retrieved_classes_emd).mode()[0] == query_class:
                emd_correct_predictions += 1

            # Precision: Proportion of correct predictions in the top 5
            true_positives_emd = sum(cls == query_class for cls in retrieved_classes_emd)
            precision_emd = true_positives_emd / top_k
            emd_precisions.append(precision_emd)
        except Exception as e:
            print(f"Error with EMD search for {query_file}: {e}")
            continue

    # Compute Global Metrics
    knn_accuracy = knn_correct_predictions / total_queries
    emd_accuracy = emd_correct_predictions / total_queries
    knn_global_precision = np.mean(knn_precisions)
    emd_global_precision = np.mean(emd_precisions)

    return {
        "k-NN": {"Accuracy": knn_accuracy, "Precision": knn_global_precision},
        "EMD": {"Accuracy": emd_accuracy, "Precision": emd_global_precision}
    }


# Main Function
def main():
    # Load feature database
    feature_file_path = "feature_vector.csv"  # Path to your feature vector file
    features_df = pd.read_csv(feature_file_path)

    # Define single-value and histogram features
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

    # Train k-NN model
    top_k = 5  # Top 5 results
    data_matrix = features_df[feature_columns].values
    knn_model = KNeighborsClassifier(n_neighbors=top_k + 1, metric='euclidean')
    knn_model.fit(data_matrix, features_df['Class'])

    # Ask user for the evaluation file path
    input_file_path = input("Enter the path to the input features CSV file: ").strip()

    # Perform evaluation
    try:
        results = evaluate_search_methods(input_file_path, knn_model, features_df, feature_columns, top_k)

        # Display Results
        print("\nEvaluation Results (Top 5 Results Only):")
        print(f"k-NN: Accuracy = {results['k-NN']['Accuracy']:.4f}, Precision = {results['k-NN']['Precision']:.4f}")
        print(f"EMD: Accuracy = {results['EMD']['Accuracy']:.4f}, Precision = {results['EMD']['Precision']:.4f}")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")


if __name__ == "__main__":
    main()
