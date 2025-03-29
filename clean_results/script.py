import pandas as pd
import os

NUMERICAL_DATA = ["higgs", "credit_card", "diabetes"]

# The datasets we have.
datasets = ["aci", "bm", "covtype", "credit_card", "diabetes", "eye_movement",
            "higgs", "kdd99"]

# List of your CSV file names
csv_files = [f"{dataset}.csv" for dataset in datasets if dataset not in NUMERICAL_DATA]

# Initialize an empty list to store individual DataFrames
dataframes = []

# Load each CSV into a DataFrame and append it to the list
for file in csv_files:
    df = pd.read_csv(file, delimiter=',')  # Assuming tab-delimited files based on column names
    dataframes.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Display or save the combined DataFrame
print(combined_df)

# Optionally save to a new CSV
combined_df.to_csv("combined_data.csv", index=False, sep='\t')
