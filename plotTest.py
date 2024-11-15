import pandas as pd

# Load the target file where D1 values will be updated
df2 = pd.read_csv('shape_descriptors.csv')

# Step 1: Build an index for df2 based on (category, model) for quick lookup
index_map = {}
for idx, row in df2.iterrows():
    key = (row['category'], row['model'])
    print(idx)
    if key not in index_map:
        index_map[key] = []
    index_map[key].append(idx)

# Step 2: Dictionary to track update counts and a list to store changes
update_counts = {}
updates = []

# Open the source file and iterate through each line, skipping the header
with open('shape_descriptors2_cleaned.csv', 'r') as source_file:
    next(source_file)  # Skip the header line

    for line in source_file:
        # Split the line into columns
        columns = line.strip().split(',')

        # Check if the line has the correct number of columns
        if len(columns) != 7:
            continue  # Skip lines with unexpected column counts

        # Extract category, model, and D1 value from this line
        category = columns[0]
        model = columns[1]
        d1_value = float(columns[3])  # D1 is the fourth column (index 3)

        # Define a unique key for each (category, model) combination
        key = (category, model)

        # Initialize the update count for this key if not already set
        if key not in update_counts:
            update_counts[key] = 0

        # Only proceed if we haven't replaced 5,000 rows for this model in df2
        if update_counts[key] < 5000 and key in index_map:
            # Get the list of matching indices for this key
            match_indices = index_map[key]

            # Ensure we don't go beyond available matches
            if update_counts[key] < len(match_indices):
                # Store the index and new D1 value for batch updating
                updates.append((match_indices[update_counts[key]], d1_value))
                update_counts[key] += 1  # Increment the count for this model
                print(update_counts[key])

# Step 3: Apply all updates to df2 in one go
for idx, new_d1_value in updates:
    df2.at[idx, 'D1'] = new_d1_value

# Save the updated DataFrame to a new CSV file
df2.to_csv('updated_file2.csv', index=False)

print("D1 values updated successfully and saved to 'updated_file2.csv'")
