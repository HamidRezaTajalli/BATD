



import pandas as pd

# Load the CSV file
file_path = '/home/htajalli/prjs0962/repos/BATD/results/eye_movement_extended.csv'
data = pd.read_csv(file_path)

# Define the grouping columns
grouping_columns = ['DATASET', 'MODEL', 'TARGET_LABEL', 'EPSILON', 'MU', 'BETA', 'LAMBDA']

# Calculate average CDA and ASR for each group
grouped = (
    data.groupby(grouping_columns, as_index=False)
    .filter(lambda x: set(range(5)).issubset(x['EXP_NUM'].unique()))
    .groupby(grouping_columns)
    .agg({'CDA': 'mean', 'ASR': 'mean'})
    .reset_index()
)

# Merge the averaged values back to the original data
merged = data.merge(grouped, on=grouping_columns, suffixes=('', '_avg'))

# Replace CDA and ASR with their averaged values in relevant rows
merged.loc[:, 'CDA'] = merged['CDA_avg']
merged.loc[:, 'ASR'] = merged['ASR_avg']

# Keep only rows with EXP_NUM = 0 and drop auxiliary columns
result = merged[merged['EXP_NUM'] == 0].drop(columns=['CDA_avg', 'ASR_avg'])

# Save the result to a new CSV file
output_file_path = '/home/htajalli/prjs0962/repos/BATD/results/processed_eye_movement_extended.csv'
result.to_csv(output_file_path, index=False)

print(f"Processed file saved to {output_file_path}")