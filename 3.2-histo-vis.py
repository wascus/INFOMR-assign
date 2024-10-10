import csv
import numpy as np
import matplotlib.pyplot as plt

# Function to read the CSV and extract the shape descriptors
def read_shape_descriptors(csv_file):
    shape_data = {}
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header

        for row in reader:
            category = row[0]
            model = row[1]
            descriptors = np.array([float(x) for x in row[2:]])  # Convert the descriptor values to floats
            shape_data[(category, model)] = descriptors

    return shape_data

# Function to visualize the histograms for a specific model using alternative graph types
def visualize_alternative_graphs(descriptors, bins, shape_name, graph_type='scatter'):
    descriptors_split = np.split(descriptors, 5)  # Split the concatenated histogram into 5 parts (A3, D1, D2, D3, D4)
    descriptor_names = ['A3 (Angle between 3 vertices)', 'D1 (Distance from barycenter to vertex)',
                        'D2 (Distance between 2 vertices)', 'D3 (Area of triangle)', 'D4 (Volume of tetrahedron)']

    # Plot each descriptor as a separate graph based on the specified graph type
    for i, descriptor in enumerate(descriptors_split):
        plt.figure()
        if graph_type == 'scatter':
            # Scatter plot
            plt.scatter(range(bins), descriptor)
            plt.title(f'{shape_name}: {descriptor_names[i]} (Scatter Plot)')
        elif graph_type == 'line':
            # Line plot
            plt.plot(range(bins), descriptor, marker='o')
            plt.title(f'{shape_name}: {descriptor_names[i]} (Line Plot)')
        elif graph_type == 'dot':
            # Dot plot
            plt.plot(range(bins), descriptor, 'bo')  # blue dots
            plt.title(f'{shape_name}: {descriptor_names[i]} (Dot Plot)')
        else:
            # Default to bar plot if no valid graph type is specified
            plt.bar(range(bins), descriptor, width=0.8)
            plt.title(f'{shape_name}: {descriptor_names[i]} (Bar Plot)')

        plt.xlabel(f'Bins (Total: {bins})')
        plt.ylabel('Frequency')
        plt.show()

# Main function to load the CSV and visualize graphs for the model(s)
def visualize_from_csv(csv_file, bins, model_name=None, graph_type='scatter'):
    # Read the descriptors from the CSV
    shape_data = read_shape_descriptors(csv_file)

    if model_name:
        # If a specific model is provided, visualize it
        for (category, model), descriptors in shape_data.items():
            if model == model_name:
                visualize_alternative_graphs(descriptors, bins, f'{category}/{model}', graph_type)
                break
        else:
            print(f'Model "{model_name}" not found in CSV.')
    else:
        # Visualize graphs for all models in the CSV
        for (category, model), descriptors in shape_data.items():
            visualize_alternative_graphs(descriptors, bins, f'{category}/{model}', graph_type)

# Main block to run the script
if __name__ == "__main__":
    csv_file = 'shape_descriptors.csv'  # Path to your CSV file
    bins = 10  # Number of bins used in the descriptor computation

    # Optionally, specify a specific model name if you only want to visualize that one model.
    # Leave it as None to visualize all models in the CSV.
    model_name = None  # Example: 'model.obj' or None for all models

    # Specify the graph type: 'scatter', 'line', 'dot', or 'bar' (default).
    graph_type = 'dot'  # Change this to 'line', 'dot', or 'bar'

    # Visualize the graphs from the CSV
    visualize_from_csv(csv_file, bins, model_name, graph_type)
