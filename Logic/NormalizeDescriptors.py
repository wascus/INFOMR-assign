import pandas as pd
from sklearn.preprocessing import RobustScaler


############################ FUNCTIONS ############################

# Robust scaling for global descriptors
# Robust scaling for global descriptors
def robust_scaling(global_descriptors):
    # Feature statistics

    stats = {
        "Surface Area": {"median": 4.1501870282463384, "iqr": 6.188773600659753},
        "Volume": {"median": 0.08746666163315656, "iqr": 0.22884952683558105},
        "Compactness": {"median": 0.01467024712758995, "iqr": 0.047229447614123996},
        "Rectangularity": {"median": 0.10759007579729485, "iqr": 0.18930044377378946},
        "Diameter": {"median": 1.8515860223664995, "iqr": 0.28410970993681195},
        "Convexity": {"median": 0.2405045856189973, "iqr": 0.37942355368166125},
        "Eccentricity": {"median": 18.14052543972729, "iqr": 42.86552768080489}
    }

    # Scale each feature using the RobustScaler formula
    scaled_descriptors = {}
    for feature, value in global_descriptors.items():
        median = stats[feature]["median"]
        iqr = stats[feature]["iqr"]
        if iqr != 0:  # Avoid division by zero
            scaled_value = (value - median) / iqr
        else:
            scaled_value = 0  # If IQR is zero, scaled value defaults to 0
        scaled_descriptors[feature] = scaled_value

    print(f"Scaled descriptor values: {scaled_descriptors}")  # Debugging log
    return scaled_descriptors


# Binning and normalizing histogram features for shape descriptors
def bin_and_normalize_histogram_features(a3_values, d1_values, d2_values, d3_values, d4_values, num_bins=40):
    histogram_features = {'A3': a3_values, 'D1': d1_values, 'D2': d2_values, 'D3': d3_values, 'D4': d4_values}
    binned_normalized_features = {}

    for feature, values in histogram_features.items():
        # Create a DataFrame for binning and normalization
        feature_df = pd.DataFrame({feature: values})
        feature_df[f'{feature}_bin'] = pd.cut(feature_df[feature], bins=num_bins, labels=False)

        # One-hot encode the bins
        binned_counts = pd.get_dummies(feature_df[f'{feature}_bin'], prefix=f'{feature}_bin')

        # Ensure all bins are present
        all_bins = [f"{feature}_bin_{i}" for i in range(num_bins)]
        binned_counts = binned_counts.reindex(columns=all_bins, fill_value=0)

        # Normalize counts within bins
        binned_counts_normalized = binned_counts.sum().div(binned_counts.sum().sum())

        # Store the normalized histogram
        binned_normalized_features[feature] = binned_counts_normalized.values.tolist()
    print(binned_normalized_features)
    return binned_normalized_features


# Normalize features and return the model's feature dictionary
def normalize_features(file_name, global_descriptors, a3_values, d1_values, d2_values, d3_values, d4_values):
    # Step 1: Scale global descriptors
    scaled_global_descriptors = robust_scaling(global_descriptors)
    print("step1 done")
    # Step 2: Normalize histogram features
    histogram_features = bin_and_normalize_histogram_features(a3_values, d1_values, d2_values, d3_values, d4_values)
    print("step2 done")
    # Step 3: Combine all features into a single dictionary
    print("step3 before")
    new_model_features = {
        "File": file_name,
        "Surface Area": scaled_global_descriptors["Surface Area"],
        "Volume": scaled_global_descriptors["Volume"],
        "Compactness": scaled_global_descriptors["Compactness"],
        "Rectangularity": scaled_global_descriptors["Rectangularity"],
        "Diameter": scaled_global_descriptors["Diameter"],
        "Convexity": scaled_global_descriptors["Convexity"],
        "Eccentricity": scaled_global_descriptors["Eccentricity"],
        "A3": histogram_features["A3"],
        "D1": histogram_features["D1"],
        "D2": histogram_features["D2"],
        "D3": histogram_features["D3"],
        "D4": histogram_features["D4"]
    }
    print(new_model_features)

    return new_model_features
