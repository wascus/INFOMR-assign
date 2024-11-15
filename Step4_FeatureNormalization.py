import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler

#load the two CSV files (global and shape property descriptors), replace file names accordingly
single_value_df = pd.read_csv('global_property_descriptors.csv', on_bad_lines='skip')
histogram_df = pd.read_csv('shape_property_descriptors.csv', on_bad_lines='skip')


############################ GLOBAL PROPERTY DESCRIPTORS ############################
#apply robust scaling standardization using sklearn
def robust_scaling(df, features):
    scaler = RobustScaler()
    df[features] = scaler.fit_transform(df[features])
    #print median and IQR for each feature, so that we can use it for normalizing new objects
    for feature, median, iqr in zip(features, scaler.center_, scaler.scale_):
        print(f"Feature: {feature}, Median: {median}, IQR: {iqr}")
    return df

#list all of the single-value features (global property descriptors), and divide per method
single_value_features = ['Surface Area', 'Volume', 'Compactness',
                         'Rectangularity', 'Diameter', 'Convexity', 'Eccentricity']

single_value_df = robust_scaling(single_value_df, single_value_features)

############################# SHAPE PROPERTY DESCRIPTORS ############################

def bin_and_normalize_histogram_features(df, num_bins=40):
    histogram_features = ['A3', 'D1', 'D2', 'D3', 'D4']
    binned_feature_dfs = []

    #bin each descriptor and normalize within each bin
    for feature in histogram_features:
        # Bin the values for each feature into specified bins
        df[f'{feature}_bin'] = pd.cut(df[feature], bins=num_bins, labels=False)

        #create a one-hot encoding of the bins in order to count the elements in each bin
        binned_counts = pd.get_dummies(df[f'{feature}_bin'], prefix=f'{feature}_bin')

        #make sure there are always 40 bins, even if all values are 0
        all_bins = [f"{feature}_bin_{i}" for i in range(num_bins)]
        binned_counts = binned_counts.reindex(columns=all_bins, fill_value=0)

        #aggregate the counts for each bin per model and normalize by dividing by the area
        binned_counts_sum = binned_counts.groupby(df['model']).sum()
        binned_counts_normalized = binned_counts_sum.div(binned_counts_sum.sum(axis=1), axis=0)

        binned_feature_dfs.append(binned_counts_normalized)

    #combine everything into a single dataframe
    combined_histogram_df = pd.concat(binned_feature_dfs, axis=1).reset_index()
    return combined_histogram_df


#normalize histogram features
normalized_histogram_df = bin_and_normalize_histogram_features(histogram_df)

############################ MERGE INTO FEATURE VECTOR ############################

#merge everything using the .obj file name ('File' in former, 'model' in latter)
merged_df = pd.merge(single_value_df, normalized_histogram_df, left_on='File', right_on='model', how='inner')

#remove some of the irrelevant columns
merged_df = merged_df.drop(columns=['model'])  #file name appears twice
merged_df = merged_df.drop(columns=['Bounding Box Volume'])  #not a relevant feature, included in file for analysis only

#save the result as one feature vector per .obj file
merged_df.to_csv('feature_vector.csv', index=False)
