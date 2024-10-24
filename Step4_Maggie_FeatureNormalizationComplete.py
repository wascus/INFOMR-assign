import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler

#load the two csv files, one with global property descriptors and the other with shape property descriptors
single_value_df = pd.read_csv('global_property_descriptors.csv', on_bad_lines='skip')
histogram_df = pd.read_csv('shape_descriptors_1.csv', on_bad_lines='skip')

############################GLOBAL PROPERTY DESCRIPTORS############################
#normalization method 1: apply robust scaling standardization using sklearn
def robust_scaling(df, features):
    scaler = RobustScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

#normalization method2: apply minmax scaling using sklearn
def min_max_scaling(df, features):
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

#list all of the single-value features (global property descriptors), and then divide per method
single_value_features = ['Surface Area', 'Volume', 'Compactness',
                         'Rectangularity', 'Diameter', 'Convexity', 'Eccentricity']

#best normalization method for each feature chosen through analyzing statistics
min_max_features = ['Diameter', 'Eccentricity']
robust_features = ['Surface Area', 'Volume', 'Compactness',
                         'Rectangularity', 'Convexity']

#apply normalization to single value features
single_value_df = min_max_scaling(single_value_df, min_max_features)
single_value_df = robust_scaling(single_value_df, robust_features)

#############################SHAPE PROPERTY DESCRIPTORS############################

def combined_normalize_histogram_features(df):
    #list all of the histogram features
    histogram_features = ['A3', 'D1', 'D2', 'D3', 'D4']

    #group by the file name ('model' column) and sum the histogram feature values for each object
    combined_histogram_df = df.groupby('model')[histogram_features].sum().reset_index()

    #normalize each histogram by dividing by the total number of elements for that object
    sample_counts = df.groupby('model').size().reset_index(name='sample_count')
    combined_histogram_df = pd.merge(combined_histogram_df, sample_counts, on='model')

    for feature in histogram_features:
        combined_histogram_df[feature] = combined_histogram_df[feature] / combined_histogram_df['sample_count']

    #remove the sample count column after normalization is complete
    combined_histogram_df = combined_histogram_df.drop(columns=['sample_count'])

    #apply minmax scaling to get values that are roughly between 0 and 1
    #without this step, a3 values were all outside this range
    scaler = MinMaxScaler()
    combined_histogram_df[histogram_features] = scaler.fit_transform(combined_histogram_df[histogram_features])

    return combined_histogram_df


#apply normalization to histogram features
normalized_histogram_df = combined_normalize_histogram_features(histogram_df)

############################MERGE INTO FEATURE VECTOR############################

#merge the single value features and the histogram features using the file name ('File' in former, 'model' in latter)
merged_df = pd.merge(single_value_df, normalized_histogram_df, left_on='File', right_on='model',
                     how='inner')

#ommitt irrelevant columns
merged_df = merged_df.drop(columns=['model']) #would make file name appear twice
merged_df = merged_df.drop(columns=['Bounding Box Volume']) #not necessary, included in file for analysis only

#save the result as one feature vector per .obj file
merged_df.to_csv('merged_normalized_features_combined.csv', index=False)
