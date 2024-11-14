import pandas as pd
import ast  # To parse the string representation of lists
from sklearn.preprocessing import RobustScaler

#load the two CSV files (global and shape property descriptors), replace file names accordingly
global_property_descriptors_df = pd.read_csv('global_property_descriptors.csv', on_bad_lines='skip')
shape_property_descriptors = pd.read_csv('shape_descriptors_pre-final.csv', on_bad_lines='skip')


############################ GLOBAL PROPERTY DESCRIPTORS ############################
#normalization method 1: apply robust scaling standardization using sklearn
def robust_scaling(df, features):
    scaler = RobustScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

#list all of the single-value features (global property descriptors), and divide per method
global_property_descriptors_list = ['Surface Area', 'Volume', 'Compactness',
                         'Rectangularity', 'Diameter', 'Convexity', 'Eccentricity']

single_value_features_normalized = robust_scaling(global_property_descriptors_df, global_property_descriptors_list)

############################# SHAPE PROPERTY DESCRIPTORS ############################

#normalize histogram bins by total element count for each feature
def normalize_histogram_bins(df, shape_property_descriptors):
    binned_feature_dfs = []

    for feature in histogram_features:
        df[feature] = df[feature].apply(lambda x: ast.literal_eval(x))  #convert values from string to list
        total_count = df[feature].apply(sum).sum() #calculate element count for each feature
        #normalize each bin value by the total count for that feature
        normalized_bins = df[feature].apply(lambda x: [v / total_count if total_count != 0 else 0 for v in x])
        #convert the list of normalized values into separate columns
        normalized_bins_df = pd.DataFrame(normalized_bins.tolist(),
                                          columns=[f'{feature}_bin_{i}' for i in range(len(df[feature][0]))])
        normalized_bins_df['model'] = df['model'] #add model identifier for referencing
        binned_feature_dfs.append(normalized_bins_df) #add to list of dataframes

    #combine everything into a single dataframe
    histogram_features_normalized = pd.concat(binned_feature_dfs, axis=1).drop_duplicates(subset=['model'])
    histogram_features_normalized = histogram_features_normalized.loc[:,
                            ~histogram_features_normalized.columns.duplicated()]  #remove duplicate columns ('model')

    return histogram_features_normalized


#list all of the histogram features
histogram_features = ['A3', 'D1', 'D2', 'D3', 'D4']
histogram_features_normalized = normalize_histogram_bins(shape_property_descriptors, histogram_features)

############################ MERGE INTO FEATURE VECTOR ############################

#merge everything using the .obj file name ('File' in former, 'model' in latter)
combined_feature_vector = pd.merge(single_value_features_normalized, histogram_features_normalized, left_on='File', right_on='model', how='inner')

#remove some of the irrelevant columns
combined_feature_vector = combined_feature_vector.drop(columns=['model'])  #file name appears twice
combined_feature_vector = combined_feature_vector.drop(columns=['Bounding Box Volume'])  #not a relevant feature, included in file for analysis only

#save the result as one feature vector per .obj file
combined_feature_vector.to_csv('feature_vector.csv', index=False)
