# INFOMR-assign

**STEP 1**  
_Step1_Open3dViewer.py:_ code for Step1, simple Viewer

**STEP 2**   
_Step2_ShapeAnalysis.py:_ code for Step2.1, analyzes a single shape in the database  
_Shape2_DatabaseStatistics.py:_ code for Step2.2, calculates shape statistics over the entire database  
_Step2_SubSamplePyMeshLab.py:_ code for Step2, processes a folder of shapes for decimation
_Step2_SuperSamplingPyMeshLab.py:_ code for Step2, processes a folder of shapes for subdivision

**STEP 3**   
_Step3_HoleFilling.py:_ hole filling using three methods  
_Step3_GlobalPropertyDescriptors.py:_ calculating the global property descriptors   
_Step3_ShapePropertyDescriptors.py:_ calculating the shape property descriptors 

_global_property_descriptors.csv:_ file containing the calculated global property descriptors  
_global_property_descriptor_statistics.xlsx:_ file containing statistics about the global property descriptors, calculated using AdditionalFiles_ComputeAverages.py   

**STEP 4**  
_Step4_FeatureNormalization.py:_ feature normalization (both single-value and histogram features), and combining everything into a feature vector  
_Step4_DistanceMatchingNew.py:_ simple version of distance matching, results are printed in the console
_Step4_DistanceMatching_GUI.py:_ interractive version of the distance matching file with a GUI   

_feature_vector.csv:_ combined feature vector with normalized single-value and histogram features  

**STEP 5**  
_Step4_DistanceMatching.py:_ simple version of KNN, produces an image with the top 5 results for a query and their distance
_Step5_KNN_GUI.py_: interractive version of the KNN search file with a GUI  
_Step5_tSNE.py_: code for performing dimensionality reduction using t-SNE   

**STEP 6**  
_Step6_Evaluation.py_: file used to evaluate the results of the queries 

**Additional Files**  
_AdditonalFIles_ComputeAverages.py:_ file that calculates averages over the entire file and per class  
