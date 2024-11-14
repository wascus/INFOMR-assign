import pandas as pd

#this file is used to compute statistics of extracted features over the entire database
#only necessary for quality control and the report - not needed to run the multimedia retrieval system

#calculate the averages of all the Global Property Descriptors, and save them as an excel file
def calculate_averages_and_save(csv_file, output_excel_file):

    df = pd.read_csv(csv_file) #read the csv (global_property_descriptors.csv)

    category_col = df.columns[0] #use the first column to get the category name
    metric_cols = df.columns[2:]  #skip the second column - it containts the .obj file name

    #calculate the average over the entire dataset
    overall_averages = df[metric_cols].mean()

    #calculate the average grouped by category (based on folder names, listed in the first column)
    category_averages = df.groupby(category_col)[metric_cols].mean()

    #write the results to excel, dividing into two sheets for convenience
    with pd.ExcelWriter(output_excel_file) as writer:
        overall_averages.to_frame(name="Overall Averages").to_excel(writer, sheet_name='Overall Averages')
        category_averages.to_excel(writer, sheet_name='Category-wise Averages')
    print(f"Averages saved to {output_excel_file}")

#substitute the following files depending on which statistics you want to calculate

#run everything - global property descriptors
#csv_file = 'global_property_descriptors.csv'  #update path as necessary
#output_excel_file = 'statistics_global_property_descriptors.xlsx'  #same here
#calculate_averages_and_save(csv_file, output_excel_file)

#run everything - merged normalized features
csv_file = 'merged_normalized_features_combined.csv'  #update path as necessary
output_excel_file = 'statistics_merged_normalized_features_combined.xlsx'  #same here
calculate_averages_and_save(csv_file, output_excel_file)
