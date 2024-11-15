# INFOMR-assign

**STEP 1**  

**STEP 2**   
_Step2_ShapeAnalysis.py:_ code for Step2.1, analyzes a single shape in the database  
_Shape2_DatabaseStatistics.py:_ code for Step2.2, calculates shape statistics over the entire database  

**STEP 3**   
_Step3_HoleFilling.py:_ hole filling using three methods  
_Step3_GlobalPropertyDescriptors.py:_ calculating the global property descriptors   

_global_property_descriptors.csv:_ file containing the calculated global property descriptors  
_global_property_descriptor_statistics.xlsx:_ file containing statistics about the global property descriptors, calculated using AdditionalFiles_ComputeAverages.py   

**STEP 4**  
_Step4_FeatureNormalization.py:_ feature normalization (both single-value and histogram features), and combining everything into a feature vector  

_feature_vector.csv:_ combined feature vector with normalized single-value and histogram features  

**STEP 5**  
_Step5_tSNE.py_: code for performing dimensionality reduction using t-SNE   

**STEP 6**  

**Additional Files**  
_AdditonalFIles_ComputeAverages.py:_ file that calculates averages over the entire file and per class  
