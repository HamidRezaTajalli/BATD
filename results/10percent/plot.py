import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

mpl.rcParams["font.size"] = 26

mu_markers = ["o", "P", "X"]
colors = ["cyan", "orange", "green", "red"]
models = ["xgboost", "tabnet", "saint", "ftt"]
epsilons = [0.01, 0.02, 0.05, 0.1]

model_map = {"xgboost": "XGBoost",
             "tabnet": "TabNet",
             "saint": "Saint",
             "ftt": "FTT"
            }

dataset_map = {"aci": "ACI",
               "bm": "BM",
               "covtype": "Forest Cover Type",
               "credit_card": "Credit Card",
               "eye_movement": "Eye Movement",
               "higgs": "HIGGS"
              }

# List of dataset names
datasets = ["aci", "bm", "covtype", "credit_card", "eye_movement", "higgs"]


def concat_data():
    """Concatenate all files to one dataframe."""

    # Read and concatenate all CSVs
    df_list = [pd.read_csv(f"{name}.csv") for name in datasets]

    # Combine into a single DataFrame
    df = pd.concat(df_list, ignore_index=True)

    return df

def plot_results(df):
    """
    This is the version of the figures we need as for every dataset we show
    the figures in one line.
    """
    # Update the model names based on model_map
    df["MODEL"] = df["MODEL"].replace(model_map)
    df["DATASET"] = df["DATASET"].replace(dataset_map)

    # Optional: Set style
    sns.set(style="whitegrid", context="talk")

    # Create the lineplot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='DATASET', y='ASR', hue='MODEL', marker='o')

    # Customize the plot
    plt.title("ASR when accessing only 10% of the data")
    plt.ylabel("ASR")
    plt.ylim(0, 110)
    plt.xlabel("Dataset")
    # Maybe we need the rotation
    plt.xticks(rotation=45)
    plt.legend(title="Model")
    plt.tight_layout()

    # Show the plot
    plt.show()

    return

def show_table(df):
    """Show the table in the appropriate format for the rebuttal."""
    df = df.drop(columns=["TARGET_LABEL", "EXP_NUM", "EPSILON", "MU", "BETA",
                          "LAMBDA", "BA_CONVERTED"])
    pivot_df = df.pivot(index='MODEL', columns='DATASET', values='ASR')
    pivot_df = pivot_df.round(2)
    pivot_df = pivot_df[['aci', 'bm', 'covtype', 'credit_card', 'higgs']]
    print(pivot_df)

    return pivot_df


if __name__ == "__main__":
    df = concat_data()

    # Clean data and keep only what we need.
    df = df[df["MU"] == 1]
    df = df[df["DATASET"] != "eye_movement"]

    # Add a column for the clean accuracy drop.
    df["drop"] = df["BA_CONVERTED"] - df["CDA"]

    _ = show_table(df)

    # Plot data
    plot_results(df)
