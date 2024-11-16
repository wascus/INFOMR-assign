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

# Example query features list: Replace this with your actual preprocessed list of query features
query_features_list = [
    {'File': 'm118.obj', 'Class': 'Humanoid', 'Surface Area': -0.3462888257829596, 'Volume': -0.11945022355544951,
     'Compactness': 0.7602443755684902, 'Rectangularity': 0.4105175803404072, 'Diameter': 0.16798907761832158,
     'Convexity': 0.3335459751751851, 'Eccentricity': 1.324402121894713, 'A3_bin_0': 0.048, 'A3_bin_1': 0.0894,
     'A3_bin_2': 0.0854, 'A3_bin_3': 0.0734, 'A3_bin_4': 0.0624, 'A3_bin_5': 0.049, 'A3_bin_6': 0.0432,
     'A3_bin_7': 0.034, 'A3_bin_8': 0.028, 'A3_bin_9': 0.0266, 'A3_bin_10': 0.023, 'A3_bin_11': 0.0174,
     'A3_bin_12': 0.0178, 'A3_bin_13': 0.019, 'A3_bin_14': 0.0176, 'A3_bin_15': 0.0168, 'A3_bin_16': 0.0134,
     'A3_bin_17': 0.0124, 'A3_bin_18': 0.0156, 'A3_bin_19': 0.012, 'A3_bin_20': 0.0148, 'A3_bin_21': 0.0122,
     'A3_bin_22': 0.0126, 'A3_bin_23': 0.0156, 'A3_bin_24': 0.0152, 'A3_bin_25': 0.015, 'A3_bin_26': 0.0136,
     'A3_bin_27': 0.0146, 'A3_bin_28': 0.017, 'A3_bin_29': 0.0174, 'A3_bin_30': 0.0186, 'A3_bin_31': 0.0178,
     'A3_bin_32': 0.0196, 'A3_bin_33': 0.0186, 'A3_bin_34': 0.019, 'A3_bin_35': 0.0168, 'A3_bin_36': 0.0172,
     'A3_bin_37': 0.0098, 'A3_bin_38': 0.0074, 'A3_bin_39': 0.0028, 'D1_bin_0': 0.0038, 'D1_bin_1': 0.0164,
     'D1_bin_2': 0.0312, 'D1_bin_3': 0.0452, 'D1_bin_4': 0.0554, 'D1_bin_5': 0.0546, 'D1_bin_6': 0.0586,
     'D1_bin_7': 0.05, 'D1_bin_8': 0.0504, 'D1_bin_9': 0.0498, 'D1_bin_10': 0.0438, 'D1_bin_11': 0.0388,
     'D1_bin_12': 0.0336, 'D1_bin_13': 0.0416, 'D1_bin_14': 0.0382, 'D1_bin_15': 0.0322, 'D1_bin_16': 0.0266,
     'D1_bin_17': 0.0348, 'D1_bin_18': 0.0292, 'D1_bin_19': 0.0318, 'D1_bin_20': 0.027, 'D1_bin_21': 0.0248,
     'D1_bin_22': 0.021, 'D1_bin_23': 0.0206, 'D1_bin_24': 0.0216, 'D1_bin_25': 0.018, 'D1_bin_26': 0.0174,
     'D1_bin_27': 0.014, 'D1_bin_28': 0.0142, 'D1_bin_29': 0.0098, 'D1_bin_30': 0.0086, 'D1_bin_31': 0.0094,
     'D1_bin_32': 0.0056, 'D1_bin_33': 0.0072, 'D1_bin_34': 0.0028, 'D1_bin_35': 0.0034, 'D1_bin_36': 0.0042,
     'D1_bin_37': 0.002, 'D1_bin_38': 0.002, 'D1_bin_39': 0.0004, 'D2_bin_0': 0.0068, 'D2_bin_1': 0.018,
     'D2_bin_2': 0.03, 'D2_bin_3': 0.0398, 'D2_bin_4': 0.0452, 'D2_bin_5': 0.0536, 'D2_bin_6': 0.0488,
     'D2_bin_7': 0.056, 'D2_bin_8': 0.0522, 'D2_bin_9': 0.0444, 'D2_bin_10': 0.045, 'D2_bin_11': 0.044,
     'D2_bin_12': 0.0418, 'D2_bin_13': 0.0418, 'D2_bin_14': 0.0356, 'D2_bin_15': 0.0298, 'D2_bin_16': 0.0318,
     'D2_bin_17': 0.0358, 'D2_bin_18': 0.031, 'D2_bin_19': 0.0258, 'D2_bin_20': 0.0268, 'D2_bin_21': 0.025,
     'D2_bin_22': 0.0218, 'D2_bin_23': 0.0188, 'D2_bin_24': 0.0152, 'D2_bin_25': 0.0114, 'D2_bin_26': 0.0138,
     'D2_bin_27': 0.0136, 'D2_bin_28': 0.012, 'D2_bin_29': 0.0118, 'D2_bin_30': 0.013, 'D2_bin_31': 0.0118,
     'D2_bin_32': 0.0106, 'D2_bin_33': 0.008, 'D2_bin_34': 0.0086, 'D2_bin_35': 0.005, 'D2_bin_36': 0.0064,
     'D2_bin_37': 0.0022, 'D2_bin_38': 0.0048, 'D2_bin_39': 0.0022, 'D3_bin_0': 0.041, 'D3_bin_1': 0.0656,
     'D3_bin_2': 0.077, 'D3_bin_3': 0.0722, 'D3_bin_4': 0.069, 'D3_bin_5': 0.07, 'D3_bin_6': 0.0632, 'D3_bin_7': 0.0614,
     'D3_bin_8': 0.0564, 'D3_bin_9': 0.056, 'D3_bin_10': 0.0438, 'D3_bin_11': 0.0384, 'D3_bin_12': 0.0418,
     'D3_bin_13': 0.03, 'D3_bin_14': 0.032, 'D3_bin_15': 0.0238, 'D3_bin_16': 0.0212, 'D3_bin_17': 0.0222,
     'D3_bin_18': 0.0172, 'D3_bin_19': 0.0168, 'D3_bin_20': 0.013, 'D3_bin_21': 0.0112, 'D3_bin_22': 0.0104,
     'D3_bin_23': 0.0086, 'D3_bin_24': 0.0076, 'D3_bin_25': 0.0052, 'D3_bin_26': 0.006, 'D3_bin_27': 0.005,
     'D3_bin_28': 0.0024, 'D3_bin_29': 0.0034, 'D3_bin_30': 0.003, 'D3_bin_31': 0.0016, 'D3_bin_32': 0.0004,
     'D3_bin_33': 0.0004, 'D3_bin_34': 0.001, 'D3_bin_35': 0.0002, 'D3_bin_36': 0.0006, 'D3_bin_37': 0.0008,
     'D3_bin_38': 0.0, 'D3_bin_39': 0.0002, 'D4_bin_0': 0.2026, 'D4_bin_1': 0.147, 'D4_bin_2': 0.1182,
     'D4_bin_3': 0.091, 'D4_bin_4': 0.08, 'D4_bin_5': 0.0678, 'D4_bin_6': 0.0524, 'D4_bin_7': 0.0402,
     'D4_bin_8': 0.0366, 'D4_bin_9': 0.0264, 'D4_bin_10': 0.0262, 'D4_bin_11': 0.0178, 'D4_bin_12': 0.0158,
     'D4_bin_13': 0.0134, 'D4_bin_14': 0.0136, 'D4_bin_15': 0.0106, 'D4_bin_16': 0.0072, 'D4_bin_17': 0.0044,
     'D4_bin_18': 0.0048, 'D4_bin_19': 0.005, 'D4_bin_20': 0.003, 'D4_bin_21': 0.0044, 'D4_bin_22': 0.0016,
     'D4_bin_23': 0.0012, 'D4_bin_24': 0.0008, 'D4_bin_25': 0.001, 'D4_bin_26': 0.0016, 'D4_bin_27': 0.0012,
     'D4_bin_28': 0.0006, 'D4_bin_29': 0.0004, 'D4_bin_30': 0.0002, 'D4_bin_31': 0.0004, 'D4_bin_32': 0.001,
     'D4_bin_33': 0.0004, 'D4_bin_34': 0.0004, 'D4_bin_35': 0.0006, 'D4_bin_36': 0.0, 'D4_bin_37': 0.0,
     'D4_bin_38': 0.0, 'D4_bin_39': 0.0002}
    ,
    {'File': '0', 'Class': 'HumanHead','Surface Area': 0.6735139810732264, 'Volume': 7.864305110528088, 'Compactness': 14.50631287097447, 'Rectangularity': 1.5849939528205845, 'Diameter': 0.4813167928784683, 'Convexity': 1.8191906300142306, 'Eccentricity': -0.38846350645540606, 'A3_bin_0': 0.0068, 'A3_bin_1': 0.0136, 'A3_bin_2': 0.0206, 'A3_bin_3': 0.0296, 'A3_bin_4': 0.0318, 'A3_bin_5': 0.0374, 'A3_bin_6': 0.0376, 'A3_bin_7': 0.0422, 'A3_bin_8': 0.0446, 'A3_bin_9': 0.0402, 'A3_bin_10': 0.0512, 'A3_bin_11': 0.0518, 'A3_bin_12': 0.0504, 'A3_bin_13': 0.0502, 'A3_bin_14': 0.0428, 'A3_bin_15': 0.051, 'A3_bin_16': 0.0524, 'A3_bin_17': 0.0482, 'A3_bin_18': 0.032, 'A3_bin_19': 0.0324, 'A3_bin_20': 0.036, 'A3_bin_21': 0.0322, 'A3_bin_22': 0.0258, 'A3_bin_23': 0.0234, 'A3_bin_24': 0.0194, 'A3_bin_25': 0.0168, 'A3_bin_26': 0.0168, 'A3_bin_27': 0.0112, 'A3_bin_28': 0.01, 'A3_bin_29': 0.01, 'A3_bin_30': 0.0076, 'A3_bin_31': 0.0056, 'A3_bin_32': 0.0048, 'A3_bin_33': 0.0046, 'A3_bin_34': 0.0032, 'A3_bin_35': 0.0018, 'A3_bin_36': 0.0016, 'A3_bin_37': 0.0014, 'A3_bin_38': 0.0006, 'A3_bin_39': 0.0004, 'D1_bin_0': 0.0002, 'D1_bin_1': 0.0008, 'D1_bin_2': 0.0016, 'D1_bin_3': 0.0032, 'D1_bin_4': 0.005, 'D1_bin_5': 0.0052, 'D1_bin_6': 0.0078, 'D1_bin_7': 0.0094, 'D1_bin_8': 0.0124, 'D1_bin_9': 0.0172, 'D1_bin_10': 0.0206, 'D1_bin_11': 0.0192, 'D1_bin_12': 0.0266, 'D1_bin_13': 0.0312, 'D1_bin_14': 0.0384, 'D1_bin_15': 0.0392, 'D1_bin_16': 0.037, 'D1_bin_17': 0.0458, 'D1_bin_18': 0.0474, 'D1_bin_19': 0.0532, 'D1_bin_20': 0.0528, 'D1_bin_21': 0.0588, 'D1_bin_22': 0.0632, 'D1_bin_23': 0.0592, 'D1_bin_24': 0.0598, 'D1_bin_25': 0.0518, 'D1_bin_26': 0.048, 'D1_bin_27': 0.036, 'D1_bin_28': 0.0358, 'D1_bin_29': 0.0276, 'D1_bin_30': 0.0214, 'D1_bin_31': 0.0198, 'D1_bin_32': 0.012, 'D1_bin_33': 0.0092, 'D1_bin_34': 0.0086, 'D1_bin_35': 0.0046, 'D1_bin_36': 0.005, 'D1_bin_37': 0.0034, 'D1_bin_38': 0.0012, 'D1_bin_39': 0.0004, 'D2_bin_0': 0.003, 'D2_bin_1': 0.005, 'D2_bin_2': 0.0074, 'D2_bin_3': 0.0096, 'D2_bin_4': 0.0124, 'D2_bin_5': 0.014, 'D2_bin_6': 0.0152, 'D2_bin_7': 0.0168, 'D2_bin_8': 0.0194, 'D2_bin_9': 0.019, 'D2_bin_10': 0.0206, 'D2_bin_11': 0.0244, 'D2_bin_12': 0.027, 'D2_bin_13': 0.025, 'D2_bin_14': 0.029, 'D2_bin_15': 0.033, 'D2_bin_16': 0.0368, 'D2_bin_17': 0.0402, 'D2_bin_18': 0.0426, 'D2_bin_19': 0.0416, 'D2_bin_20': 0.0424, 'D2_bin_21': 0.0534, 'D2_bin_22': 0.0578, 'D2_bin_23': 0.0614, 'D2_bin_24': 0.058, 'D2_bin_25': 0.0578, 'D2_bin_26': 0.05, 'D2_bin_27': 0.0438, 'D2_bin_28': 0.0338, 'D2_bin_29': 0.0242, 'D2_bin_30': 0.0202, 'D2_bin_31': 0.0146, 'D2_bin_32': 0.0126, 'D2_bin_33': 0.0094, 'D2_bin_34': 0.0048, 'D2_bin_35': 0.0056, 'D2_bin_36': 0.0038, 'D2_bin_37': 0.0026, 'D2_bin_38': 0.0008, 'D2_bin_39': 0.001, 'D3_bin_0': 0.0164, 'D3_bin_1': 0.0214, 'D3_bin_2': 0.0306, 'D3_bin_3': 0.0348, 'D3_bin_4': 0.0334, 'D3_bin_5': 0.0386, 'D3_bin_6': 0.0406, 'D3_bin_7': 0.0404, 'D3_bin_8': 0.034, 'D3_bin_9': 0.0368, 'D3_bin_10': 0.0384, 'D3_bin_11': 0.0384, 'D3_bin_12': 0.0442, 'D3_bin_13': 0.0392, 'D3_bin_14': 0.0388, 'D3_bin_15': 0.0434, 'D3_bin_16': 0.0356, 'D3_bin_17': 0.0392, 'D3_bin_18': 0.0392, 'D3_bin_19': 0.0376, 'D3_bin_20': 0.0388, 'D3_bin_21': 0.0354, 'D3_bin_22': 0.0304, 'D3_bin_23': 0.0312, 'D3_bin_24': 0.0226, 'D3_bin_25': 0.0238, 'D3_bin_26': 0.0236, 'D3_bin_27': 0.0196, 'D3_bin_28': 0.0138, 'D3_bin_29': 0.0098, 'D3_bin_30': 0.0074, 'D3_bin_31': 0.0074, 'D3_bin_32': 0.0058, 'D3_bin_33': 0.0024, 'D3_bin_34': 0.0034, 'D3_bin_35': 0.0006, 'D3_bin_36': 0.0018, 'D3_bin_37': 0.0008, 'D3_bin_38': 0.0002, 'D3_bin_39': 0.0002, 'D4_bin_0': 0.1266, 'D4_bin_1': 0.101, 'D4_bin_2': 0.0798, 'D4_bin_3': 0.0698, 'D4_bin_4': 0.0596, 'D4_bin_5': 0.0584, 'D4_bin_6': 0.0532, 'D4_bin_7': 0.0478, 'D4_bin_8': 0.0468, 'D4_bin_9': 0.0382, 'D4_bin_10': 0.0344, 'D4_bin_11': 0.0346, 'D4_bin_12': 0.0256, 'D4_bin_13': 0.0244, 'D4_bin_14': 0.0204, 'D4_bin_15': 0.0194, 'D4_bin_16': 0.0226, 'D4_bin_17': 0.0158, 'D4_bin_18': 0.0164, 'D4_bin_19': 0.0132, 'D4_bin_20': 0.0094, 'D4_bin_21': 0.0136, 'D4_bin_22': 0.0098, 'D4_bin_23': 0.0088, 'D4_bin_24': 0.0072, 'D4_bin_25': 0.008, 'D4_bin_26': 0.0068, 'D4_bin_27': 0.0058, 'D4_bin_28': 0.005, 'D4_bin_29': 0.004, 'D4_bin_30': 0.0038, 'D4_bin_31': 0.002, 'D4_bin_32': 0.002, 'D4_bin_33': 0.0014, 'D4_bin_34': 0.0018, 'D4_bin_35': 0.0006, 'D4_bin_36': 0.0012, 'D4_bin_37': 0.0002, 'D4_bin_38': 0.0004, 'D4_bin_39': 0.0002}

    ,
    {'File': 'D01017.obj', 'Class': 'Chess', 'Surface Area': 0.41636159507867604, 'Volume': 3.4574635036847416,
     'Compactness': 5.763326861239068, 'Rectangularity': 1.1530133435670076, 'Diameter': 0.23593736112381342,
     'Convexity': 0.8109270040791013, 'Eccentricity': -0.3058929079166295, 'A3_bin_0': 0.0134, 'A3_bin_1': 0.0348,
     'A3_bin_2': 0.0362, 'A3_bin_3': 0.0478, 'A3_bin_4': 0.0504, 'A3_bin_5': 0.051, 'A3_bin_6': 0.0514,
     'A3_bin_7': 0.0518, 'A3_bin_8': 0.0482, 'A3_bin_9': 0.0378, 'A3_bin_10': 0.038, 'A3_bin_11': 0.0402,
     'A3_bin_12': 0.0378, 'A3_bin_13': 0.0378, 'A3_bin_14': 0.0364, 'A3_bin_15': 0.0364, 'A3_bin_16': 0.035,
     'A3_bin_17': 0.0272, 'A3_bin_18': 0.0308, 'A3_bin_19': 0.0282, 'A3_bin_20': 0.0272, 'A3_bin_21': 0.0268,
     'A3_bin_22': 0.0214, 'A3_bin_23': 0.0196, 'A3_bin_24': 0.0168, 'A3_bin_25': 0.0156, 'A3_bin_26': 0.0166,
     'A3_bin_27': 0.0114, 'A3_bin_28': 0.0104, 'A3_bin_29': 0.0102, 'A3_bin_30': 0.0094, 'A3_bin_31': 0.0084,
     'A3_bin_32': 0.0074, 'A3_bin_33': 0.0056, 'A3_bin_34': 0.0068, 'A3_bin_35': 0.0042, 'A3_bin_36': 0.0048,
     'A3_bin_37': 0.0028, 'A3_bin_38': 0.0024, 'A3_bin_39': 0.0016, 'D1_bin_0': 0.0004, 'D1_bin_1': 0.002,
     'D1_bin_2': 0.005, 'D1_bin_3': 0.0082, 'D1_bin_4': 0.0106, 'D1_bin_5': 0.017, 'D1_bin_6': 0.0206,
     'D1_bin_7': 0.0292, 'D1_bin_8': 0.0312, 'D1_bin_9': 0.038, 'D1_bin_10': 0.0484, 'D1_bin_11': 0.0466,
     'D1_bin_12': 0.0446, 'D1_bin_13': 0.0452, 'D1_bin_14': 0.0456, 'D1_bin_15': 0.0478, 'D1_bin_16': 0.0376,
     'D1_bin_17': 0.0426, 'D1_bin_18': 0.0388, 'D1_bin_19': 0.038, 'D1_bin_20': 0.0326, 'D1_bin_21': 0.0372,
     'D1_bin_22': 0.032, 'D1_bin_23': 0.0298, 'D1_bin_24': 0.0278, 'D1_bin_25': 0.0272, 'D1_bin_26': 0.026,
     'D1_bin_27': 0.0258, 'D1_bin_28': 0.0266, 'D1_bin_29': 0.023, 'D1_bin_30': 0.0212, 'D1_bin_31': 0.024,
     'D1_bin_32': 0.0188, 'D1_bin_33': 0.0158, 'D1_bin_34': 0.0106, 'D1_bin_35': 0.0096, 'D1_bin_36': 0.0068,
     'D1_bin_37': 0.0052, 'D1_bin_38': 0.002, 'D1_bin_39': 0.0006, 'D2_bin_0': 0.0036, 'D2_bin_1': 0.0048,
     'D2_bin_2': 0.0098, 'D2_bin_3': 0.0132, 'D2_bin_4': 0.017, 'D2_bin_5': 0.0184, 'D2_bin_6': 0.0244,
     'D2_bin_7': 0.0238, 'D2_bin_8': 0.0312, 'D2_bin_9': 0.0334, 'D2_bin_10': 0.0408, 'D2_bin_11': 0.0402,
     'D2_bin_12': 0.047, 'D2_bin_13': 0.0392, 'D2_bin_14': 0.0412, 'D2_bin_15': 0.0452, 'D2_bin_16': 0.0414,
     'D2_bin_17': 0.0382, 'D2_bin_18': 0.0388, 'D2_bin_19': 0.0358, 'D2_bin_20': 0.0344, 'D2_bin_21': 0.0334,
     'D2_bin_22': 0.0242, 'D2_bin_23': 0.0244, 'D2_bin_24': 0.0266, 'D2_bin_25': 0.0258, 'D2_bin_26': 0.022,
     'D2_bin_27': 0.029, 'D2_bin_28': 0.0204, 'D2_bin_29': 0.0216, 'D2_bin_30': 0.0204, 'D2_bin_31': 0.0234,
     'D2_bin_32': 0.022, 'D2_bin_33': 0.0172, 'D2_bin_34': 0.022, 'D2_bin_35': 0.017, 'D2_bin_36': 0.0122,
     'D2_bin_37': 0.0112, 'D2_bin_38': 0.0044, 'D2_bin_39': 0.001, 'D3_bin_0': 0.024, 'D3_bin_1': 0.0476,
     'D3_bin_2': 0.0528, 'D3_bin_3': 0.061, 'D3_bin_4': 0.0642, 'D3_bin_5': 0.0628, 'D3_bin_6': 0.0636,
     'D3_bin_7': 0.0638, 'D3_bin_8': 0.058, 'D3_bin_9': 0.0502, 'D3_bin_10': 0.054, 'D3_bin_11': 0.0468,
     'D3_bin_12': 0.0428, 'D3_bin_13': 0.0346, 'D3_bin_14': 0.0378, 'D3_bin_15': 0.0286, 'D3_bin_16': 0.0276,
     'D3_bin_17': 0.0214, 'D3_bin_18': 0.0258, 'D3_bin_19': 0.0156, 'D3_bin_20': 0.0202, 'D3_bin_21': 0.0122,
     'D3_bin_22': 0.0174, 'D3_bin_23': 0.0102, 'D3_bin_24': 0.0112, 'D3_bin_25': 0.0084, 'D3_bin_26': 0.0068,
     'D3_bin_27': 0.0074, 'D3_bin_28': 0.0044, 'D3_bin_29': 0.0038, 'D3_bin_30': 0.0052, 'D3_bin_31': 0.002,
     'D3_bin_32': 0.0018, 'D3_bin_33': 0.002, 'D3_bin_34': 0.0022, 'D3_bin_35': 0.0002, 'D3_bin_36': 0.0008,
     'D3_bin_37': 0.0004, 'D3_bin_38': 0.0, 'D3_bin_39': 0.0004, 'D4_bin_0': 0.1956, 'D4_bin_1': 0.1372,
     'D4_bin_2': 0.1072, 'D4_bin_3': 0.0898, 'D4_bin_4': 0.0658, 'D4_bin_5': 0.0592, 'D4_bin_6': 0.0586,
     'D4_bin_7': 0.0444, 'D4_bin_8': 0.0288, 'D4_bin_9': 0.0292, 'D4_bin_10': 0.0248, 'D4_bin_11': 0.0236,
     'D4_bin_12': 0.0218, 'D4_bin_13': 0.0156, 'D4_bin_14': 0.015, 'D4_bin_15': 0.011, 'D4_bin_16': 0.0114,
     'D4_bin_17': 0.011, 'D4_bin_18': 0.0094, 'D4_bin_19': 0.0046, 'D4_bin_20': 0.0064, 'D4_bin_21': 0.0042,
     'D4_bin_22': 0.0034, 'D4_bin_23': 0.0032, 'D4_bin_24': 0.0026, 'D4_bin_25': 0.0022, 'D4_bin_26': 0.003,
     'D4_bin_27': 0.0018, 'D4_bin_28': 0.0016, 'D4_bin_29': 0.0012, 'D4_bin_30': 0.0022, 'D4_bin_31': 0.0008,
     'D4_bin_32': 0.0002, 'D4_bin_33': 0.001, 'D4_bin_34': 0.0006, 'D4_bin_35': 0.0002, 'D4_bin_36': 0.0002,
     'D4_bin_37': 0.0004, 'D4_bin_38': 0.0004, 'D4_bin_39': 0.0004},
    {'File': 'D00023.obj', 'Class': 'Guitar', 'Surface Area': -0.3225259983628649, 'Volume': -0.19056708678661702,
     'Compactness': 0.1501292135575952, 'Rectangularity': 0.694188753832657, 'Diameter': -0.13549222310780365,
     'Convexity': 0.6501359506747836, 'Eccentricity': 7.646256246487903, 'A3_bin_0': 0.1572, 'A3_bin_1': 0.0964,
     'A3_bin_2': 0.0762, 'A3_bin_3': 0.0514, 'A3_bin_4': 0.0432, 'A3_bin_5': 0.0358, 'A3_bin_6': 0.0282,
     'A3_bin_7': 0.0258, 'A3_bin_8': 0.0218, 'A3_bin_9': 0.0188, 'A3_bin_10': 0.0156, 'A3_bin_11': 0.014,
     'A3_bin_12': 0.0146, 'A3_bin_13': 0.0128, 'A3_bin_14': 0.0122, 'A3_bin_15': 0.0106, 'A3_bin_16': 0.0118,
     'A3_bin_17': 0.0124, 'A3_bin_18': 0.0118, 'A3_bin_19': 0.0136, 'A3_bin_20': 0.0124, 'A3_bin_21': 0.0112,
     'A3_bin_22': 0.0128, 'A3_bin_23': 0.0144, 'A3_bin_24': 0.0096, 'A3_bin_25': 0.0144, 'A3_bin_26': 0.0134,
     'A3_bin_27': 0.0102, 'A3_bin_28': 0.0142, 'A3_bin_29': 0.014, 'A3_bin_30': 0.015, 'A3_bin_31': 0.016,
     'A3_bin_32': 0.0174, 'A3_bin_33': 0.0202, 'A3_bin_34': 0.0208, 'A3_bin_35': 0.0262, 'A3_bin_36': 0.0278,
     'A3_bin_37': 0.023, 'A3_bin_38': 0.016, 'A3_bin_39': 0.0068, 'D1_bin_0': 0.0082, 'D1_bin_1': 0.023,
     'D1_bin_2': 0.0296, 'D1_bin_3': 0.0332, 'D1_bin_4': 0.0314, 'D1_bin_5': 0.0364, 'D1_bin_6': 0.037,
     'D1_bin_7': 0.0358, 'D1_bin_8': 0.0358, 'D1_bin_9': 0.037, 'D1_bin_10': 0.0334, 'D1_bin_11': 0.0362,
     'D1_bin_12': 0.0344, 'D1_bin_13': 0.032, 'D1_bin_14': 0.0342, 'D1_bin_15': 0.0332, 'D1_bin_16': 0.0328,
     'D1_bin_17': 0.0296, 'D1_bin_18': 0.03, 'D1_bin_19': 0.0346, 'D1_bin_20': 0.0284, 'D1_bin_21': 0.036,
     'D1_bin_22': 0.0348, 'D1_bin_23': 0.0338, 'D1_bin_24': 0.0376, 'D1_bin_25': 0.0326, 'D1_bin_26': 0.0376,
     'D1_bin_27': 0.035, 'D1_bin_28': 0.0294, 'D1_bin_29': 0.0222, 'D1_bin_30': 0.013, 'D1_bin_31': 0.013,
     'D1_bin_32': 0.0026, 'D1_bin_33': 0.0014, 'D1_bin_34': 0.0012, 'D1_bin_35': 0.0014, 'D1_bin_36': 0.0012,
     'D1_bin_37': 0.0004, 'D1_bin_38': 0.0002, 'D1_bin_39': 0.0004, 'D2_bin_0': 0.0386, 'D2_bin_1': 0.064,
     'D2_bin_2': 0.0706, 'D2_bin_3': 0.06, 'D2_bin_4': 0.0442, 'D2_bin_5': 0.034, 'D2_bin_6': 0.0342, 'D2_bin_7': 0.029,
     'D2_bin_8': 0.0256, 'D2_bin_9': 0.02, 'D2_bin_10': 0.0212, 'D2_bin_11': 0.016, 'D2_bin_12': 0.0154,
     'D2_bin_13': 0.015, 'D2_bin_14': 0.011, 'D2_bin_15': 0.0102, 'D2_bin_16': 0.0128, 'D2_bin_17': 0.015,
     'D2_bin_18': 0.0186, 'D2_bin_19': 0.0182, 'D2_bin_20': 0.0222, 'D2_bin_21': 0.0206, 'D2_bin_22': 0.0284,
     'D2_bin_23': 0.023, 'D2_bin_24': 0.0282, 'D2_bin_25': 0.0238, 'D2_bin_26': 0.02, 'D2_bin_27': 0.0244,
     'D2_bin_28': 0.033, 'D2_bin_29': 0.035, 'D2_bin_30': 0.041, 'D2_bin_31': 0.034, 'D2_bin_32': 0.0338,
     'D2_bin_33': 0.0248, 'D2_bin_34': 0.016, 'D2_bin_35': 0.0072, 'D2_bin_36': 0.0046, 'D2_bin_37': 0.0022,
     'D2_bin_38': 0.003, 'D2_bin_39': 0.0012, 'D3_bin_0': 0.2212, 'D3_bin_1': 0.1436, 'D3_bin_2': 0.1156,
     'D3_bin_3': 0.0872, 'D3_bin_4': 0.0828, 'D3_bin_5': 0.0706, 'D3_bin_6': 0.044, 'D3_bin_7': 0.0374,
     'D3_bin_8': 0.028, 'D3_bin_9': 0.0216, 'D3_bin_10': 0.0158, 'D3_bin_11': 0.0188, 'D3_bin_12': 0.0184,
     'D3_bin_13': 0.0152, 'D3_bin_14': 0.0124, 'D3_bin_15': 0.0094, 'D3_bin_16': 0.0106, 'D3_bin_17': 0.0106,
     'D3_bin_18': 0.0082, 'D3_bin_19': 0.0054, 'D3_bin_20': 0.0034, 'D3_bin_21': 0.0034, 'D3_bin_22': 0.0036,
     'D3_bin_23': 0.003, 'D3_bin_24': 0.0016, 'D3_bin_25': 0.002, 'D3_bin_26': 0.0012, 'D3_bin_27': 0.0004,
     'D3_bin_28': 0.0008, 'D3_bin_29': 0.0006, 'D3_bin_30': 0.0002, 'D3_bin_31': 0.0008, 'D3_bin_32': 0.0004,
     'D3_bin_33': 0.0002, 'D3_bin_34': 0.0006, 'D3_bin_35': 0.0004, 'D3_bin_36': 0.0004, 'D3_bin_37': 0.0,
     'D3_bin_38': 0.0, 'D3_bin_39': 0.0002, 'D4_bin_0': 0.4714, 'D4_bin_1': 0.1534, 'D4_bin_2': 0.0904,
     'D4_bin_3': 0.0576, 'D4_bin_4': 0.0424, 'D4_bin_5': 0.0328, 'D4_bin_6': 0.0214, 'D4_bin_7': 0.021,
     'D4_bin_8': 0.0198, 'D4_bin_9': 0.0122, 'D4_bin_10': 0.0132, 'D4_bin_11': 0.0092, 'D4_bin_12': 0.009,
     'D4_bin_13': 0.0094, 'D4_bin_14': 0.0038, 'D4_bin_15': 0.0058, 'D4_bin_16': 0.0034, 'D4_bin_17': 0.004,
     'D4_bin_18': 0.0038, 'D4_bin_19': 0.0028, 'D4_bin_20': 0.003, 'D4_bin_21': 0.0012, 'D4_bin_22': 0.0006,
     'D4_bin_23': 0.0022, 'D4_bin_24': 0.0016, 'D4_bin_25': 0.0006, 'D4_bin_26': 0.0012, 'D4_bin_27': 0.0006,
     'D4_bin_28': 0.0004, 'D4_bin_29': 0.0006, 'D4_bin_30': 0.0002, 'D4_bin_31': 0.0, 'D4_bin_32': 0.0002,
     'D4_bin_33': 0.0004, 'D4_bin_34': 0.0002, 'D4_bin_35': 0.0, 'D4_bin_36': 0.0, 'D4_bin_37': 0.0, 'D4_bin_38': 0.0,
     'D4_bin_39': 0.0002}
    ,
    {'File': 'D00078.obj', 'Class': 'Bicycle', 'Surface Area': -0.20719590738338767, 'Volume': -0.2948075962502034,
     'Compactness': -0.2700085134927901, 'Rectangularity': -0.5074826835906896, 'Diameter': 0.36527218447032,
     'Convexity': -0.5454894201946817, 'Eccentricity': 0.4499631912039982, 'A3_bin_0': 0.0422, 'A3_bin_1': 0.0496,
     'A3_bin_2': 0.0518, 'A3_bin_3': 0.057, 'A3_bin_4': 0.0524, 'A3_bin_5': 0.0482, 'A3_bin_6': 0.0472,
     'A3_bin_7': 0.043, 'A3_bin_8': 0.0342, 'A3_bin_9': 0.0388, 'A3_bin_10': 0.0302, 'A3_bin_11': 0.0368,
     'A3_bin_12': 0.0308, 'A3_bin_13': 0.027, 'A3_bin_14': 0.0328, 'A3_bin_15': 0.0268, 'A3_bin_16': 0.0274,
     'A3_bin_17': 0.0266, 'A3_bin_18': 0.0224, 'A3_bin_19': 0.026, 'A3_bin_20': 0.021, 'A3_bin_21': 0.0194,
     'A3_bin_22': 0.0186, 'A3_bin_23': 0.0196, 'A3_bin_24': 0.0164, 'A3_bin_25': 0.0168, 'A3_bin_26': 0.0164,
     'A3_bin_27': 0.015, 'A3_bin_28': 0.013, 'A3_bin_29': 0.012, 'A3_bin_30': 0.0124, 'A3_bin_31': 0.0098,
     'A3_bin_32': 0.0088, 'A3_bin_33': 0.0126, 'A3_bin_34': 0.0102, 'A3_bin_35': 0.0086, 'A3_bin_36': 0.0058,
     'A3_bin_37': 0.0058, 'A3_bin_38': 0.0044, 'A3_bin_39': 0.0022, 'D1_bin_0': 0.0024, 'D1_bin_1': 0.0056,
     'D1_bin_2': 0.0114, 'D1_bin_3': 0.0186, 'D1_bin_4': 0.025, 'D1_bin_5': 0.032, 'D1_bin_6': 0.037,
     'D1_bin_7': 0.0446, 'D1_bin_8': 0.0458, 'D1_bin_9': 0.0456, 'D1_bin_10': 0.0436, 'D1_bin_11': 0.0536,
     'D1_bin_12': 0.0422, 'D1_bin_13': 0.049, 'D1_bin_14': 0.0478, 'D1_bin_15': 0.044, 'D1_bin_16': 0.0472,
     'D1_bin_17': 0.0504, 'D1_bin_18': 0.0362, 'D1_bin_19': 0.0412, 'D1_bin_20': 0.0374, 'D1_bin_21': 0.0312,
     'D1_bin_22': 0.0256, 'D1_bin_23': 0.0282, 'D1_bin_24': 0.027, 'D1_bin_25': 0.0168, 'D1_bin_26': 0.023,
     'D1_bin_27': 0.0158, 'D1_bin_28': 0.0126, 'D1_bin_29': 0.0118, 'D1_bin_30': 0.0118, 'D1_bin_31': 0.011,
     'D1_bin_32': 0.007, 'D1_bin_33': 0.0068, 'D1_bin_34': 0.0034, 'D1_bin_35': 0.0034, 'D1_bin_36': 0.0014,
     'D1_bin_37': 0.0012, 'D1_bin_38': 0.0012, 'D1_bin_39': 0.0002, 'D2_bin_0': 0.0068, 'D2_bin_1': 0.0178,
     'D2_bin_2': 0.023, 'D2_bin_3': 0.0182, 'D2_bin_4': 0.0188, 'D2_bin_5': 0.0196, 'D2_bin_6': 0.0442,
     'D2_bin_7': 0.0416, 'D2_bin_8': 0.0386, 'D2_bin_9': 0.0322, 'D2_bin_10': 0.0408, 'D2_bin_11': 0.0428,
     'D2_bin_12': 0.0444, 'D2_bin_13': 0.0396, 'D2_bin_14': 0.0432, 'D2_bin_15': 0.0468, 'D2_bin_16': 0.0476,
     'D2_bin_17': 0.0416, 'D2_bin_18': 0.0346, 'D2_bin_19': 0.0352, 'D2_bin_20': 0.0294, 'D2_bin_21': 0.0326,
     'D2_bin_22': 0.0346, 'D2_bin_23': 0.0318, 'D2_bin_24': 0.0324, 'D2_bin_25': 0.03, 'D2_bin_26': 0.0216,
     'D2_bin_27': 0.016, 'D2_bin_28': 0.0154, 'D2_bin_29': 0.017, 'D2_bin_30': 0.014, 'D2_bin_31': 0.014,
     'D2_bin_32': 0.0088, 'D2_bin_33': 0.007, 'D2_bin_34': 0.0042, 'D2_bin_35': 0.0048, 'D2_bin_36': 0.0034,
     'D2_bin_37': 0.0024, 'D2_bin_38': 0.002, 'D2_bin_39': 0.0012, 'D3_bin_0': 0.1092, 'D3_bin_1': 0.1098,
     'D3_bin_2': 0.1114, 'D3_bin_3': 0.0882, 'D3_bin_4': 0.0742, 'D3_bin_5': 0.0618, 'D3_bin_6': 0.0604,
     'D3_bin_7': 0.0518, 'D3_bin_8': 0.0494, 'D3_bin_9': 0.0412, 'D3_bin_10': 0.0352, 'D3_bin_11': 0.0338,
     'D3_bin_12': 0.0318, 'D3_bin_13': 0.0232, 'D3_bin_14': 0.0204, 'D3_bin_15': 0.0154, 'D3_bin_16': 0.0132,
     'D3_bin_17': 0.0136, 'D3_bin_18': 0.0092, 'D3_bin_19': 0.0092, 'D3_bin_20': 0.0072, 'D3_bin_21': 0.0046,
     'D3_bin_22': 0.004, 'D3_bin_23': 0.004, 'D3_bin_24': 0.0028, 'D3_bin_25': 0.0022, 'D3_bin_26': 0.002,
     'D3_bin_27': 0.0014, 'D3_bin_28': 0.0012, 'D3_bin_29': 0.0012, 'D3_bin_30': 0.001, 'D3_bin_31': 0.001,
     'D3_bin_32': 0.0012, 'D3_bin_33': 0.0012, 'D3_bin_34': 0.0012, 'D3_bin_35': 0.0004, 'D3_bin_36': 0.0006,
     'D3_bin_37': 0.0, 'D3_bin_38': 0.0002, 'D3_bin_39': 0.0002, 'D4_bin_0': 0.4646, 'D4_bin_1': 0.1944,
     'D4_bin_2': 0.108, 'D4_bin_3': 0.0748, 'D4_bin_4': 0.0454, 'D4_bin_5': 0.0314, 'D4_bin_6': 0.0224,
     'D4_bin_7': 0.015, 'D4_bin_8': 0.0116, 'D4_bin_9': 0.0068, 'D4_bin_10': 0.005, 'D4_bin_11': 0.0054,
     'D4_bin_12': 0.0036, 'D4_bin_13': 0.0034, 'D4_bin_14': 0.0018, 'D4_bin_15': 0.0016, 'D4_bin_16': 0.0008,
     'D4_bin_17': 0.0004, 'D4_bin_18': 0.0002, 'D4_bin_19': 0.0006, 'D4_bin_20': 0.0002, 'D4_bin_21': 0.0006,
     'D4_bin_22': 0.0008, 'D4_bin_23': 0.0002, 'D4_bin_24': 0.0, 'D4_bin_25': 0.0002, 'D4_bin_26': 0.0004,
     'D4_bin_27': 0.0002, 'D4_bin_28': 0.0, 'D4_bin_29': 0.0, 'D4_bin_30': 0.0, 'D4_bin_31': 0.0, 'D4_bin_32': 0.0,
     'D4_bin_33': 0.0, 'D4_bin_34': 0.0, 'D4_bin_35': 0.0, 'D4_bin_36': 0.0, 'D4_bin_37': 0.0, 'D4_bin_38': 0.0,
     'D4_bin_39': 0.0002}

]


def compute_distance(query, candidate, histogram_features, single_value_features, feature_weights):
    """
    Compute the weighted EMD and Euclidean distance between two shapes.
    """
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
    """
    Query the K-nearest neighbors from the database for a given query shape.
    """
    distances = []
    for _, row in database_df.iterrows():
        dist = compute_distance(query_features, row, histogram_features, single_value_features, feature_weights)
        distances.append((row['File'], row['Class'], dist))
    distances.sort(key=lambda x: x[2])
    return distances[:top_k]


def evaluate_query(query_features, database_df, histogram_features, single_value_features, feature_weights, top_k=10):
    """
    Evaluate precision and accuracy for a single query.
    """
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

    overall_precision = []
    overall_accuracy = []

    for query_features in query_features_list:
        query_df = pd.DataFrame([query_features])  # Convert query to DataFrame row

        # Evaluate query
        precision, accuracy = evaluate_query(query_df.iloc[0], database_df, histogram_features, single_value_features,
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
