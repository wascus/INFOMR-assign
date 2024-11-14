import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable

#read the excel file
df = pd.read_excel('shape_analysis_final.xlsx')

#calculate statistics for each shape - mean, standard deviation, min and max
avg_faces = df['num_faces'].mean()
std_faces = df['num_faces'].std()
min_faces = df['num_faces'].min()
max_faces = df['num_faces'].max()
avg_vertices = df['num_vertices'].mean()
std_vertices = df['num_vertices'].std()
min_vertices = df['num_vertices'].min()
max_vertices = df['num_vertices'].max()

#save the results in a df
def create_statistics_summary(df):
    statistics_summary = pd.DataFrame({
        "Statistic": ["Mean", "Standard Deviation", "Minimum", "Maximum"],
        "Faces": [avg_faces, std_faces, min_faces, max_faces],
        "Vertices": [avg_vertices, std_vertices, min_vertices, max_vertices]
    })
    return statistics_summary

statistics_summary = create_statistics_summary(df)

#plot a histogram with the number of faces
plt.hist(df['num_faces'], bins=20, color='blue', alpha=0.7)
plt.title('Histogram of Number of Faces')
plt.xlabel('Number of Faces')
plt.ylabel('Frequency')
plt.show()

#plot a histogram with the number of vertices
plt.hist(df['num_vertices'], bins=20, color='green', alpha=0.7)
plt.title('Histogram of Number of Vertices')
plt.xlabel('Number of Vertices')
plt.ylabel('Frequency')
plt.show()

#identify outliers (shapes with many or few vertices/faces)
outliers = df[(df['num_vertices'] < 100) | (df['num_vertices'] > 35000) |
              (df['num_faces'] < 100) | (df['num_faces'] > 50000)]

#write the results to excel, with a sheet for statistics and a sheet for outliers
with pd.ExcelWriter('shape_analysis_stats_and_outliers_final.xlsx') as writer:
    statistics_summary.to_excel(writer, sheet_name='Statistics', index=False)
    if not outliers.empty:
        outliers.to_excel(writer, sheet_name='Outliers', index=False)

