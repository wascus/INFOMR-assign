import matplotlib
matplotlib.use('TkAgg')  #TkAgg backend allows us to use interactive plots
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the normalized features data
file_path = "feature_vector.csv"
features_df = pd.read_csv(file_path)

# Extract features and labels
feature_columns = [
    "Surface Area", "Volume", "Compactness", "Rectangularity", "Diameter",
    "Convexity", "Eccentricity",
    "A3_bin_0.0", "A3_bin_1.0", "A3_bin_2.0", "A3_bin_3.0", "A3_bin_4.0", "A3_bin_5.0", "A3_bin_6.0", "A3_bin_7.0", "A3_bin_8.0", "A3_bin_9.0",
    "D1_bin_0.0", "D1_bin_1.0", "D1_bin_2.0", "D1_bin_3.0", "D1_bin_4.0", "D1_bin_5.0", "D1_bin_6.0", "D1_bin_7.0", "D1_bin_8.0", "D1_bin_9.0",
    "D2_bin_0.0", "D2_bin_1.0", "D2_bin_2.0", "D2_bin_3.0", "D2_bin_4.0", "D2_bin_5.0", "D2_bin_6.0", "D2_bin_7.0", "D2_bin_8.0", "D2_bin_9.0",
    "D3_bin_0.0", "D3_bin_1.0", "D3_bin_2.0", "D3_bin_3.0", "D3_bin_4.0", "D3_bin_5.0", "D3_bin_6.0", "D3_bin_7.0", "D3_bin_8.0", "D3_bin_9.0",
    "D4_bin_0.0", "D4_bin_1.0", "D4_bin_2.0", "D4_bin_3.0", "D4_bin_4.0", "D4_bin_5.0", "D4_bin_6.0", "D4_bin_7.0", "D4_bin_8.0", "D4_bin_9.0"
]

# Filter to only include the first 69 classes
unique_classes = features_df['Class'].unique()
limited_classes = unique_classes[:69]
filtered_df = features_df[features_df['Class'].isin(limited_classes)]

# Extract features for the filtered DataFrame
X = filtered_df[feature_columns].values
y = filtered_df['Class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform dimensionality reduction using t-SNE
tsne = TSNE(n_components=2, perplexity=35, learning_rate=500, max_iter=1000, random_state=42)
features_2d = tsne.fit_transform(X_scaled)

# Combine colors from tab20, tab20b, and tab20c to ensure enough distinct colors
colormap = plt.get_cmap('tab20').colors + plt.get_cmap('tab20b').colors + plt.get_cmap('tab20c').colors + plt.get_cmap('Set3').colors
colors = colormap[:69]  # Use only the first 69 colors

# Map each class to a color
class_color_map = {cls: colors[i] for i, cls in enumerate(limited_classes)}

# Plot 2D scatterplot with class colors
plt.figure(figsize=(16, 12))
scatter = plt.scatter(
    features_2d[:, 0], features_2d[:, 1],
    c=[class_color_map[cls] for cls in filtered_df['Class']],
    alpha=0.7
)

# Create legend with class names and custom colors
handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_name,
                      markerfacecolor=class_color_map[class_name], markersize=8)
           for class_name in limited_classes]

# Set the legend in a single column and adjust font size
plt.legend(handles=handles, title='Class', bbox_to_anchor=(1, 1), loc='upper left', ncol=2, fontsize=8)

# Remove x and y axis ticks
plt.xticks([])  # Remove x-axis ticks
plt.yticks([])  # Remove y-axis ticks

# Annotate points with class and filename on hover
annot = plt.annotate("", xy=(0, 0), xytext=(20, 20),
                     textcoords="offset points",
                     bbox=dict(boxstyle="round", fc="w"),
                     arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    pos = scatter.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = f"Class: {filtered_df.iloc[ind['ind'][0]]['Class']}\nFile: {filtered_df.iloc[ind['ind'][0]]['File']}"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.8)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == plt.gca():
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            plt.draw()
        else:
            if vis:
                annot.set_visible(False)
                plt.draw()

plt.gcf().canvas.mpl_connect("motion_notify_event", hover)
plt.show()
