import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

#plt.rcParams.update({"font.size": 26})
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

# KDD and Diabetes are not very trustworthy datasets.
datasets = ["aci", "bm", "credit_card", "diabetes", "eye_movement", "covtype",
            "higgs"]

def best_mu(dataset, model, target_label, data):
    df = data[(data["DATASET"] == dataset) & (data["MODEL"] == model) &
              (data["TARGET_LABEL"] == target_label)]

    cnt = [0, 0, 0]
    mu_index_map ={0.2: 0, 0.5: 1, 1: 2}
    for epsilon in epsilons:
        tmp = df[df["EPSILON"] == epsilon].sort_values(by="ASR")
        cnt[mu_index_map[tmp.iloc[2]["MU"]]] += 1

    if cnt[2] >= 2:
        return 1.0
    elif np.argmax(cnt) == 0:
        return 0.2
    elif np.argmax(cnt) == 1:
        return 0.5


def find_mus():
    """
    Find the mus that give the best accuracy for each model and target
    label.
    """
    mapping = {}
    for dataset in datasets:

        # Load the CSV file
        data = pd.read_csv(f"{dataset}.csv", delimiter=",")

        # Ensure the required columns are present
        required_columns = ["EXP_NUM", "DATASET", "MODEL", "TARGET_LABEL",
                "EPSILON", "MU", "BETA", "LAMBDA", "BA_CONVERTED", "CDA",
                "ASR"]
        if not all(column in data.columns for column in required_columns):
            raise ValueError("The CSV file must contain the following columns:"
                             " " + ", ".join(required_columns))

        mapping[dataset] = {}
        for model in models:
            mus = []
            for target_label in data["TARGET_LABEL"].unique():
                mus.append(best_mu(dataset, model, target_label, data))
            mapping[dataset][model] = mus
    return mapping


def v1(mapping):
    """Plot the first version of our figures."""
    for dataset in datasets:
        # Load the CSV file
        data = pd.read_csv(f"{dataset}.csv", delimiter=",")

        # Ensure the required columns are present
        required_columns = ["EXP_NUM", "DATASET", "MODEL", "TARGET_LABEL",
                "EPSILON", "MU", "BETA", "LAMBDA", "BA_CONVERTED", "CDA", "ASR"]
        if not all(column in data.columns for column in required_columns):
            raise ValueError("The CSV file must contain the following columns: "
                             + ", ".join(required_columns))

        # Get unique models
        models = data["MODEL"].unique()

        # Create a plot for each model
        for model in models:
            print(f"Dataset: {dataset}, Model: {model}")
            model_data = data[data["MODEL"] == model]

            plt.figure(figsize=(10, 6))

            # Plot lines for each TARGET_LABEL
            for target_label in model_data["TARGET_LABEL"].unique():
                good_mu = mapping[dataset][model][target_label]
                label_data = model_data[(model_data["TARGET_LABEL"] == target_label) &
                                        (model_data["MU"] == good_mu)]\
                                       .sort_values(by="EPSILON")
                plt.plot(label_data["EPSILON"] * 100, label_data["ASR"],
                         marker='o', label=f"Target Label {target_label}")

            # Add plot details
            plt.title(f"Dataset: {dataset}, Model: {model}")
            plt.xlabel("Poisoning Rate (%)")
            plt.ylabel("ASR (%)")
            plt.legend(title="Target Label")
            plt.grid(True)

            # Save or show the plot
            plt.savefig(f"{dataset}_{model}_v1.png")
            plt.show()
            plt.close()


def v2():
    """This is the second version of the figures we need."""
    for dataset in datasets:

        if dataset == "eye_movement":
            data = pd.read_csv("eye_movement_averaged.csv", delimiter=",")
        else:
            # Load the CSV file
            data = pd.read_csv(f"{dataset}.csv", delimiter=",")

        # Ensure the required columns are present
        required_columns = ["EXP_NUM", "DATASET", "MODEL", "TARGET_LABEL",
                "EPSILON", "MU", "BETA", "LAMBDA", "BA_CONVERTED", "CDA", "ASR"]
        if not all(column in data.columns for column in required_columns):
            raise ValueError("The CSV file must contain the following columns: "
                             + ", ".join(required_columns))

        # Get unique models
        models = data["MODEL"].unique()

        # Create a plot for each model
        for model in models:
            print(f"Dataset: {dataset}, Model: {model}")
            model_data = data[data["MODEL"] == model]

            plt.figure(figsize=(8, 5))

            # Plot lines for each TARGET_LABEL
            for target_label in model_data["TARGET_LABEL"].unique():
                for i, mu in enumerate(model_data["MU"].unique()):
                    label_data = model_data[(model_data["TARGET_LABEL"] == target_label) &
                                            (model_data["MU"] == mu)]\
                                           .sort_values(by="EPSILON")
                    plt.plot(label_data["EPSILON"] * 100, label_data["ASR"],
                             marker=mu_markers[i], linestyle="-",
                             color=f"C{target_label}",
                             markersize=12, linewidth=2,
                             label=f"{target_label}/{mu}"
                    )
                plt.plot(label_data["EPSILON"] * 100, label_data["CDA"],
                         marker=mu_markers[0], linestyle="--",
                         color=f"C{target_label}", markersize=12,
                         linewidth=2
                )

            # Add plot details
            plt.title(f"Dataset: {dataset}, Model: {model}")
            plt.xlabel("Poisoning Rate (%)")
            plt.ylabel("Performance (%)")
            plt.grid(True)

            ax = plt.gca()
            ax.set_ylim([0, 100])

            plt.subplots_adjust(left=0.16, bottom=0.18, right=0.9, top=0.88,
                                wspace=0.2, hspace=0.2)

            custom_lines = [
                    Line2D([0], [0], color="black", linestyle="-", label="ASR"),
                    Line2D([0], [0], color="black", linestyle="--", label="CDA"),
            ]
            legend1 = plt.legend(custom_lines, ["ASR", "CDA"],
                                 loc="upper left",
                                 bbox_to_anchor=(1.2, 0.12),
                                 title="Quantity"
            )
            plt.gca().add_artist(legend1)
            legend2 = plt.legend(title="Target Label/Mu",
                                 loc="upper right",
                                 bbox_to_anchor=(1.6, 1.1),
                                 ncol=2
            )

            #plt.subplots_adjust(left=0.07, bottom=0.1, right=0.43, top=0.6,
            #                    wspace=0.5, hspace=0.66)
            plt.savefig(f"{dataset}_{model}_v2.png")
            plt.show()
            plt.close()

    return

def v3():
    """This is the second version of the figures we need."""
    for dataset in datasets:

        if dataset == "eye_movement":
            data = pd.read_csv("eye_movement_averaged.csv", delimiter=",")
        else:
            # Load the CSV file
            data = pd.read_csv(f"{dataset}.csv", delimiter=",")

        # Ensure the required columns are present
        required_columns = ["EXP_NUM", "DATASET", "MODEL", "TARGET_LABEL",
                "EPSILON", "MU", "BETA", "LAMBDA", "BA_CONVERTED", "CDA", "ASR"]
        if not all(column in data.columns for column in required_columns):
            raise ValueError("The CSV file must contain the following columns: "
                             + ", ".join(required_columns))

        # Get unique models
        models = data["MODEL"].unique()

        # Create a plot for each model
        for model in models:
            print(f"Dataset: {dataset}, Model: {model}")
            model_data = data[data["MODEL"] == model]

            plt.figure(figsize=(6, 4))

            # Plot lines for each TARGET_LABEL
            for target_label in model_data["TARGET_LABEL"].unique():
                for i, mu in enumerate(model_data["MU"].unique()):
                    label_data = model_data[(model_data["TARGET_LABEL"] == target_label) &
                                            (model_data["MU"] == mu)]\
                                           .sort_values(by="EPSILON")
                    plt.plot(label_data["EPSILON"] * 100, label_data["ASR"],
                             marker=mu_markers[i], linestyle="-",
                             color=f"C{target_label}",
                             markersize=12, linewidth=2,
                             label=f"{target_label}/{mu}"
                    )
                plt.plot(label_data["EPSILON"] * 100, label_data["CDA"],
                         marker=mu_markers[0], linestyle="--",
                         color=f"C{target_label}", markersize=12,
                         linewidth=2
                )

            # Add plot details
            plt.title(f"Model: {model_map[model]}")
            plt.xlabel("Poisoning Rate (%)")
            plt.ylabel("Performance (%)")
            plt.grid(True)

            ax = plt.gca()
            ax.set_ylim([0, 100])

            plt.subplots_adjust(left=0.23, bottom=0.24, right=0.88, top=0.88,
                                wspace=0.2, hspace=0.2)

            custom_lines = [
                    Line2D([0], [0], color="black", linestyle="-", label="ASR"),
                    Line2D([0], [0], color="black", linestyle="--", label="CDA"),
            ]
            legend1 = plt.legend(custom_lines, ["ASR", "CDA"],
                                 loc="upper left",
                                 bbox_to_anchor=(1.2, 0.12),
                                 title="Quantity"
            )
            plt.gca().add_artist(legend1)
            legend2 = plt.legend(title="Target Label/Mu",
                                 loc="upper right",
                                 bbox_to_anchor=(1.6, 1.1),
                                 ncol=2
            )

            plt.subplots_adjust(left=0.07, bottom=0.1, right=0.43, top=0.6,
                                wspace=0.5, hspace=0.66)
            plt.savefig(f"{dataset}_{model}_v3.png")
            plt.show()
            plt.close()

    return


def latex_table():
    """Print the values from all the files in a latex table way."""
    # Read and concatenate data from all files
    data_frames = [
            pd.read_csv(f"{dataset}.csv") if "eye" not in dataset
            else pd.read_csv(f"{dataset}_averaged.csv")
            for dataset in datasets
    ]
    combined_data = pd.concat(data_frames, ignore_index=True)

    # Filter the data
    df = combined_data[(combined_data["MU"] == 1)]


    # Define EPSILON values to include in the summary
    epsilon_values = sorted(df["EPSILON"].unique())[:4]


    # Create a summary DataFrame
    summary_data = []
    for (dataset, model, target_label), group in df.groupby(["DATASET", "MODEL", "TARGET_LABEL"]):
        row = {"DATASET": dataset, "MODEL": model, "TARGET_LABEL": target_label}
        for epsilon in epsilon_values:
            subset = group[group["EPSILON"] == epsilon]
            if not subset.empty:
                cda = subset["CDA"].iloc[0]
                asr = subset["ASR"].iloc[0]
                row[f"EPSILON_{epsilon}"] = f"{cda:.2f}/{asr:.2f}"
            else:
                row[f"EPSILON_{epsilon}"] = "N/A"
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    pd.set_option('display.max_rows', None)
    print(summary_df)

    return

def credit_tabnet():
    """
    Create a figure for credit card dataset and TabNet as we now have
    multiple executions of the experiment for label 1.
    """
    dataset = "credit_card"

    # Load the CSV file
    data = pd.read_csv(f"{dataset}.csv", delimiter=",")

    # Get unique models
    models = data["MODEL"].unique()
    model = "tabnet"

    print(f"Dataset: {dataset}, Model: {model}")
    model_data = data[data["MODEL"] == model]

    plt.figure(figsize=(25, 15))

    # Plot lines for each TARGET_LABEL
    for target_label in model_data["TARGET_LABEL"].unique():
        for i, mu in enumerate(model_data["MU"].unique()):
            label_data = model_data[(model_data["TARGET_LABEL"] == target_label) &
                                    (model_data["MU"] == mu)]\
                                   .sort_values(by="EPSILON")
            if target_label == 1:
                label_data = label_data.groupby(["EPSILON"]).mean().reset_index()

            plt.plot(label_data["EPSILON"] * 100, label_data["ASR"],
                     marker=mu_markers[i], linestyle="-",
                     color=f"C{target_label}",
                     markersize=12, linewidth=2,
                     label=f"{target_label}/{mu}"
            )
        plt.plot(label_data["EPSILON"] * 100, label_data["CDA"],
                 marker=mu_markers[0], linestyle="--",
                 color=f"C{target_label}", markersize=12,
                 linewidth=2
        )

    # Add plot details
    plt.title(f"Dataset: {dataset}, Model: {model}")
    plt.xlabel("Poisoning Rate (%)")
    plt.ylabel("Performance (%)")
    plt.grid(True)

    custom_lines = [
            Line2D([0], [0], color="black", linestyle="-", label="ASR"),
            Line2D([0], [0], color="black", linestyle="--", label="CDA"),
    ]
    legend1 = plt.legend(custom_lines, ["ASR", "CDA"],
                         loc="upper left",
                         bbox_to_anchor=(1.06, 0.1),
                         title="Quantity"
    )
    plt.gca().add_artist(legend1)
    legend2 = plt.legend(title="Target Label/Mu",
                         loc="upper right",
                         bbox_to_anchor=(1.3, 1)
    )

    ax = plt.gca()
    ax.set_ylim([0, 100])

    plt.subplots_adjust(right=0.75, top=0.95, bottom=0.10, hspace=0.66,
                        left=0.16, wspace=0.5)
    plt.savefig(f"{dataset}_{model}_v2.png")
    #plt.show()
    plt.close()

    return

def v4():
    """
    This is the version of the figures we need as for every dataset we show
    the figures in one line.
    """
    for dataset in datasets:

        if dataset == "eye_movement":
            data = pd.read_csv("eye_movement_averaged.csv", delimiter=",")
        elif dataset == "credit_card":
            data = pd.read_csv("credit_card_all.csv", delimiter=",")
        else:
            # Load the CSV file
            data = pd.read_csv(f"{dataset}.csv", delimiter=",")

        # Ensure the required columns are present
        required_columns = ["EXP_NUM", "DATASET", "MODEL", "TARGET_LABEL",
                "EPSILON", "MU", "BETA", "LAMBDA", "BA_CONVERTED", "CDA", "ASR"]
        if not all(column in data.columns for column in required_columns):
            raise ValueError("The CSV file must contain the following columns: "
                             + ", ".join(required_columns))

        # Get unique models
        models = data["MODEL"].unique()

        # Create one figure with 4 subfigures per dataset
        fig, axes = plt.subplots(
                1, 4, figsize=(22, 5.6), sharex=True, sharey=True
        )
        axes = axes.flatten()

        for i, model in enumerate(models[:4]):  # Limit to 4 models per dataset
            print(f"Dataset: {dataset}, Model: {model}")
            model_data = data[data["MODEL"] == model]
            ax = axes[i]

            # Plot lines for each TARGET_LABEL
            for target_label in model_data["TARGET_LABEL"].unique():
                for j, mu in enumerate(model_data["MU"].unique()):
                    label_data = model_data[(model_data["TARGET_LABEL"] == target_label) &
                                            (model_data["MU"] == mu)]\
                                           .sort_values(by="EPSILON")
                    if ((target_label == 1) and (model == "tabnet") and
                       (dataset == "credit_card")):
                        label_data = label_data.groupby(["EPSILON"])\
                                               .mean(numeric_only=True)\
                                               .reset_index()
                    ax.plot(label_data["EPSILON"] * 100, label_data["ASR"],
                            marker=mu_markers[j], linestyle="-",
                            color=f"C{target_label}",
                            markersize=14, linewidth=2,
                            label=f"{target_label}/{mu}"
                    )
                ax.plot(label_data["EPSILON"] * 100, label_data["CDA"],
                        linestyle="--", color=f"C{target_label}",
                        #markersize=14, marker=mu_markers[0],
                        linewidth=2
                )

            # Add subplot details
            ax.set_title(f"{model_map[model]}")
            ax.grid(True)
            ax.set_ylim([0, 105])

        # Set common labels in the middle of the figure
        fig.text(0.5, 0.04, "Poisoning Rate (%)", ha="center", fontsize=26)
        fig.text(0.01, 0.5, "Performance (%)", va="center", rotation="vertical", fontsize=26)
        fig.subplots_adjust(left=0.08, bottom=0.21, right=0.98, top=0.86,
                            wspace=0.11, hspace=0.2)

        #plt.tight_layout()
        plt.savefig(f"{dataset}_models_v4.png")
        #plt.show()
        plt.close()

    return


def only_legends():
    """This is the second version of the figures we need."""
    for dataset in datasets:

        if dataset == "eye_movement":
            data = pd.read_csv("eye_movement_averaged.csv", delimiter=",")
        else:
            # Load the CSV file
            data = pd.read_csv(f"{dataset}.csv", delimiter=",")

        # Ensure the required columns are present
        required_columns = ["EXP_NUM", "DATASET", "MODEL", "TARGET_LABEL",
                "EPSILON", "MU", "BETA", "LAMBDA", "BA_CONVERTED", "CDA", "ASR"]
        if not all(column in data.columns for column in required_columns):
            raise ValueError("The CSV file must contain the following columns: "
                             + ", ".join(required_columns))

        # Get unique models
        models = data["MODEL"].unique()

        # Create one figure with 4 subfigures per dataset
        fig, axes = plt.subplots(
                1, 4, figsize=(18.5, 4), sharex=True, sharey=True
        )
        axes = axes.flatten()

        for i, model in enumerate(models[:4]):  # Limit to 4 models per dataset
            print(f"Dataset: {dataset}, Model: {model}")
            model_data = data[data["MODEL"] == model]
            ax = axes[i]

            # Plot lines for each TARGET_LABEL
            for target_label in model_data["TARGET_LABEL"].unique():
                for j, mu in enumerate(model_data["MU"].unique()):
                    label_data = model_data[(model_data["TARGET_LABEL"] == target_label) &
                                            (model_data["MU"] == mu)]\
                                           .sort_values(by="EPSILON")
                    if ((target_label == 1) and (model == "tabnet") and
                       (dataset == "credit_card")):
                        label_data = label_data.groupby(["EPSILON"])\
                                               .mean(numeric_only=True)\
                                               .reset_index()
                    #ax.plot(label_data["EPSILON"] * 100, label_data["ASR"],
                    ax.plot([], [],
                            marker=mu_markers[j], linestyle="-",
                            color=f"C{target_label}",
                            markersize=14, linewidth=2,
                            label=f"{target_label}/{mu}"
                    )
                #ax.plot(label_data["EPSILON"] * 100, label_data["CDA"],
                ax.plot([], [],
                        marker=mu_markers[0], linestyle="--",
                        color=f"C{target_label}", markersize=14,
                        linewidth=2
                )

            # Create a dictionary with the mapping from dataset to positions
            # and number of columns for the legend.
            positions = {
                    "aci": [(1.3, 0.6), (4, 1.1), 6],
                    "bm": [(1.3, 0.6), (4, 1.1), 6],
                    "diabetes": [(1.3, 0.6), (4, 1.1), 6],
                    "credit_card": [(1.3, 0.6), (4, 1.1), 6],
                    "eye_movement": [(1.3, 0.4), (3.6, 1.1), 5],
                    "covtype": [(1.35, 0.25), (4.42, 1.2), 7],
                    "higgs": [(1.3, 0.6), (4, 1.1), 6],
            }

            show_legend = True
            if show_legend and i == 0:
                l1_pos, l2_pos, l2_cols = positions[dataset]
                custom_lines = [
                        Line2D([0], [0], color="black", linestyle="-", label="ASR"),
                        Line2D([0], [0], color="black", linestyle="--", label="CDA"),
                ]
                legend1 = ax.legend(custom_lines, ["ASR", "CDA"],
                                     loc="upper left",
                                     bbox_to_anchor=l1_pos,
                                     title="Quantity",
                                     ncol=2
                )
                ax.add_artist(legend1)
                legend2 = ax.legend(title="Target Label/$\mu$",
                                     loc="upper right",
                                     bbox_to_anchor=l2_pos,
                                     ncol=l2_cols
                )
                ax.set_frame_on(False)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
            elif show_legend and i > 0:
                ax.set_frame_on(False)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

        fig.subplots_adjust(left=0.08, bottom=0.21, right=0.98, top=0.84,
                            wspace=0.11, hspace=0.2)

        #plt.tight_layout()
        plt.savefig(f"{dataset}_legend_v4.png")
        #plt.show()
        plt.close()

    return


if __name__ == "__main__":
    #v1(find_mus())
    #v2()
    #latex_table()
    #v3()
    #credit_tabnet()
    v4()
    only_legends()
