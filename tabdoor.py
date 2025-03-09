import numpy as np
import torch
import logging
import time # Import time module for Unix timestamp
from pathlib import Path
import csv
from torch.utils.data import TensorDataset
from attack import Attack

# importing all the models for this project
from models.FTT import FTTModel
from models.Tabnet import TabNetModel
from models.SAINT import SAINTModel
from models.CatBoost import CatBoostModel
from models.XGBoost import XGBoostModel

# importing all the datasets for this project
from dataset.BM import BankMarketing
from dataset.ACI import ACI
from dataset.HIGGS import HIGGS
from dataset.Diabetes import Diabetes
from dataset.Eye_Movement import EyeMovement
from dataset.KDD99 import KDD99
from dataset.CreditCard import CreditCard
from dataset.CovType import CovType  # Importing the ConvertCovType class from CovType.py
import math






# ============================================
# Logging Configuration
# ============================================

# Set up the logging configuration to display informational messages.
# This helps in tracking the progress and debugging if necessary.
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the logging message format
    handlers=[
        logging.StreamHandler()  # Output logs to the console
    ]
)





# ============================================
# Main Execution Block
# ============================================


def attack_step_by_step(args, use_saved_models, use_existing_trigger):
    """
    The main function orchestrates the attack setup by executing the initial model training steps,
    retraining on the poisoned dataset, and evaluating the backdoor's effectiveness.
    """

    dataset_name = args.dataset_name
    model_name = args.model_name
    target_label = args.target_label
    epsilon = args.epsilon
    exp_num = args.exp_num


    models_path = Path("./saved_models/tabdoor")

    # create the experiment results directory
    results_path = Path("./results/tabdoor")
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    csv_file_address = results_path / Path(f"{dataset_name}.csv")
    if not csv_file_address.exists():
        csv_file_address.touch()
        
        csv_header = ['EXP_NUM', 'DATASET', 'MODEL', 'TARGET_LABEL', 'EPSILON', 'CDA', 'ASR']
        # insert the header row into the csv file
        with open(csv_file_address, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)
        



    dataset_dict = {
        "aci": ACI,
        "bm": BankMarketing,
        "higgs": HIGGS,
        "diabetes": Diabetes,
        "eye_movement": EyeMovement,
        "kdd99": KDD99,
        "credit_card": CreditCard,
        "covtype": CovType
    }

    model_dict = {
        "ftt": FTTModel,
        "tabnet": TabNetModel,
        "saint": SAINTModel,
        "catboost": CatBoostModel,
        "xgboost": XGBoostModel
    }

    top_3_features = {
        "bm": ['previous', 'duration', 'day'],
        "aci": ['Capital Gain', 'Capital Loss', 'Education-Num'],
        "higgs": ['m_bb', 'm_wwbb', 'm_wbb'],
        "eye_movement": ['regressDur', 'nWordsInTitle', 'nRegressFrom'],
        "credit_card": ['V14', 'V10', 'V7'],
        "covtype": ['Elevation', 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points']
    }




    logging.info(f"Performing BadNet Attack with the following args: {args}")

    # Step 0: Initialize device, target label, and other parameters needed for the attack.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Step 1: Initialize the dataset object which can handle, convert and revert the dataset.
    data_obj = dataset_dict[dataset_name]()

    # Step 2: Initialize the model. If needed (optional), the model can be loaded from a saved model. Then the model is not needed to be trained again.

    if model_name == "ftt":
        model = FTTModel(data_obj=data_obj)
    elif model_name == "tabnet":
        model = TabNetModel(
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            momentum=0.3,
            mask_type='entmax'
        )
    elif model_name == "saint":
        model = SAINTModel(data_obj=data_obj, is_numerical=False)

    elif model_name == "xgboost" or model_name == "catboost":
        model = model_dict[model_name](objective="multi" if data_obj.num_classes > 2 else "binary")



    if model_name == "ftt" and data_obj.cat_cols:
        model.to(device, model_type="original")
 
    model.to(device)


    # Step 3: Initialize the Attack class with model, data, and target label
    attack = Attack(device=device, model=model, data_obj=data_obj, target_label=target_label, mu=0, beta=0, lambd=0, epsilon=epsilon)




    # Step 11: Revert the poisoned dataset to the original categorical features
    FTT = True if attack.model.model_name == "FTTransformer" else False

    # get the clean train and test datasets
    
    clean_dataset = attack.data_obj.get_normal_datasets()
    clean_trainset = clean_dataset[0]
    clean_testset = clean_dataset[1]


    features_to_poison = top_3_features[dataset_name]

    trigger_dictionary = {}

    for feature in features_to_poison:
        feature_idx = data_obj.column_idx[feature]
        if feature in data_obj.cat_cols:
            raise ValueError(f"Feature {feature} is a categorical feature. Please choose a numerical feature.")
        else:
            # Use the mode (most common value) of the feature
            # For continuous features, we need to bin the values first to find the most common value
            # Using pandas value_counts() to find the most frequent value
            mode_value = float(data_obj.X_encoded[feature].value_counts().idxmax())  # Convert to float to ensure compatibility with tensor
            trigger_dictionary[feature] = mode_value

    print("the trigger dictionary is: \n", trigger_dictionary)

    

    ftt_split = True if FTT and data_obj.cat_cols else False

    poisoned_trainset = poison_dataset(data_obj, clean_trainset, trigger_dictionary, epsilon=args.epsilon, target_label=target_label, ftt_split=ftt_split)
    poisoned_testset = poison_dataset(data_obj, clean_testset, trigger_dictionary, epsilon= 1, target_label=target_label, ftt_split=ftt_split)
    poisoned_dataset = (poisoned_trainset, poisoned_testset)



    logging.info("=== Training the model on the poisoned training dataset ===")
    # Step 12: Train the model on the poisoned training dataset
    converted = False if FTT and data_obj.cat_cols else True
    attack.train(poisoned_dataset, converted=converted)
    logging.info("=== Poisoned Training Completed ===")

    logging.info("=== Testing the model on the poisoned testing dataset and clean testing dataset ===")
    # Step 13: Test the model on the poisoned testing dataset and clean testing dataset
    if FTT and data_obj.cat_cols:
        clean_testset = attack.data_obj.get_normal_datasets_FTT()[1]

    asr = attack.test(poisoned_testset, converted=converted)
    cda = attack.test(clean_testset, converted=converted)
    logging.info("=== Testing Completed ===")

    # Step 14: Save the results to the csv file
    with open(csv_file_address, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([exp_num, dataset_name, model_name, target_label, epsilon, cda, asr])


    # creating and checking the path for saving and loading the poisoned models.
    poisoned_model_path = models_path / Path(f"poisoned")
    if not poisoned_model_path.exists():
        poisoned_model_path.mkdir(parents=True, exist_ok=True)
    poisoned_model_address = poisoned_model_path / Path(f"{model_name}_{data_obj.dataset_name}_{target_label}_{epsilon}_poisoned_model_td.pth")

    # Save the poisoned model with Unix timestamp in the filename
    if model_name == "ftt" and data_obj.cat_cols:
        attack.model.save_model(poisoned_model_address, model_type="original")
    else:
        attack.model.save_model(poisoned_model_address)

    # save the trigger dictionary

    trigger_path = Path("./saved_triggers/tabdoor")
    if not trigger_path.exists():
        trigger_path.mkdir(parents=True, exist_ok=True)
    trigger_address = trigger_path / Path(f"{model_name}_{data_obj.dataset_name}_{target_label}_{epsilon}_trigger_dictionary.pt")
    torch.save(trigger_dictionary, trigger_address)

    




def poison_dataset(data_obj, dataset, trigger_dictionary, epsilon, target_label, ftt_split):

    # Pick args.epsilon number of samples from the clean trainset randomly to poison

    X_tensor, y_tensor = dataset.tensors

    num_samples_to_poison = int(len(X_tensor) * epsilon)
    indices_to_poison = np.random.choice(len(X_tensor), num_samples_to_poison, replace=False)


    # Create a poisoned dataset by applying the trigger values
    poisoned_X_tensor = X_tensor.clone()  
    poisoned_y_tensor = y_tensor.clone()
    for idx in indices_to_poison:
        for feature, trigger_value in trigger_dictionary.items():
            feature_idx = data_obj.column_idx[feature]
            poisoned_X_tensor[idx][feature_idx] = trigger_value
        poisoned_y_tensor[idx] = torch.tensor(target_label, dtype=torch.long)
    if ftt_split:
        poisoned_X_tensor_categorical = poisoned_X_tensor[:, data_obj.cat_cols_idx].to(torch.long)
        poisoned_X_tensor_numerical = poisoned_X_tensor[:, data_obj.num_cols_idx].to(torch.float32)
        poisoned_dataset = TensorDataset(poisoned_X_tensor_categorical, poisoned_X_tensor_numerical, poisoned_y_tensor)
    else:
        poisoned_dataset = TensorDataset(poisoned_X_tensor, poisoned_y_tensor)
    
    return poisoned_dataset




# ============================================


if __name__ == "__main__":
    # set up an argument parser
    import argparse
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--exp_num", type=int, default=0)

    # parse the arguments
    args = parser.parse_args()

    available_datasets = ["aci", "bm", "higgs", "diabetes", "eye_movement", "kdd99", "credit_card", "covtype"]
    available_models = ["ftt", "tabnet", "saint", "catboost", "xgboost"]

    # check and make sure that the dataset and model are available
    # the dataset and model names are case insensitive, so compare the lowercase versions of the arguments
    if args.dataset_name.lower() not in available_datasets:
        raise ValueError(f"Dataset {args.dataset_name} is not available. Please choose from: {available_datasets}")
    if args.model_name.lower() not in available_models:
        raise ValueError(f"Model {args.model_name} is not available. Please choose from: {available_models}")

     
     # check and make sure that mu, beta, lambd, and epsilon are within the valid range
    if args.epsilon < 0 or args.epsilon > 1:
        raise ValueError(f"Epsilon must be between 0 and 1. You provided: {args.epsilon}")
    
    use_saved_models = {'clean': False, 'poisoned': False}
    use_existing_trigger = False

    # log all the arguments before running the experiment
    print("-"*100)
    logging.info(f"Running experiment with the following arguments: {args}")
    print("-"*100)
    print("\n")



    attack_step_by_step(args, use_saved_models, use_existing_trigger)


