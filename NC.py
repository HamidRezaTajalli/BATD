from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from attack import Attack
from nc_files import reverse_trigger, mad_outlier_detection



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




###############################################################################
# 5. Main function
###############################################################################
if __name__ == "__main__":
    """
    In practice, you would:
      1) Train or load your model (TabNet, SAINT, or FT-Transformer).
      2) Extract penultimate features via the appropriate function.
      3) Run spectral_signature_defense(...) to identify suspicious samples.
      4) If you have a ground-truth poison mask, or if you want to treat
         suspicious indices as "poison", you can plot with plotCorrelationScores.
    """


    # set up an argument parser
    import argparse
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lambd", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--exp_num", type=int, default=0)

    # parse the arguments
    args = parser.parse_args()

    available_datasets = ["aci", "bm", "higgs", "diabetes", "eye_movement", "kdd99", "credit_card", "covtype"]
    available_models = ["ftt", "tabnet", "saint"]

    # check and make sure that the dataset and model are available
    # the dataset and model names are case insensitive, so compare the lowercase versions of the arguments
    if args.dataset_name.lower() not in available_datasets:
        raise ValueError(f"Dataset {args.dataset_name} is not available. Please choose from: {available_datasets}")
    if args.model_name.lower() not in available_models:
        raise ValueError(f"Model {args.model_name} is not available. Please choose from: {available_models}")

     
     # check and make sure that mu, beta, lambd, and epsilon are within the valid range
    if args.mu < 0 or args.mu > 1:
        raise ValueError(f"Mu must be between 0 and 1. You provided: {args.mu}")
    if args.beta < 0 or args.beta > 1:
        raise ValueError(f"Beta must be between 0 and 1. You provided: {args.beta}")
    if args.lambd < 0 or args.lambd > 1:
        raise ValueError(f"Lambd must be between 0 and 1. You provided: {args.lambd}")
    if args.epsilon < 0 or args.epsilon > 1:
        raise ValueError(f"Epsilon must be between 0 and 1. You provided: {args.epsilon}")
    
    

    # log all the arguments before running the experiment
    print("-"*100)
    logging.info(f"Neural Cleanse with the following arguments: {args}")
    print("-"*100)
    print("\n")

    model_name = args.model_name
    dataset_name = args.dataset_name
    target_label = args.target_label
    mu = args.mu
    beta = args.beta
    lambd = args.lambd
    epsilon = args.epsilon


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


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Step 1: Initialize the dataset object which can handle, convert and revert the dataset.
    data_obj = dataset_dict[dataset_name]()


    models_path = Path("./saved_models")

    # creating and checking the path for saving and loading the poisoned models.
    poisoned_model_path = models_path / Path(f"poisoned")
    if not poisoned_model_path.exists():
        poisoned_model_path.mkdir(parents=True, exist_ok=True)
    poisoned_model_address = poisoned_model_path / Path(f"{model_name}_{data_obj.dataset_name}_{target_label}_{mu}_{beta}_{lambd}_{epsilon}_poisoned_model.pth")


    # if the model name is xgboost, catboost, or tabnet, then we should add .zip at the end of the model name.
    if model_name in ['tabnet', 'xgboost', 'catboost']:
        # add .zip at the end of the model name in addition to the existing suffix
        poisoned_model_address_toload = poisoned_model_address.with_suffix(poisoned_model_address.suffix + ".zip")
    else:
        poisoned_model_address_toload = poisoned_model_address

    if not poisoned_model_address_toload.exists():
        raise ValueError(f"Poisoned model address at {poisoned_model_address_toload} does not exist.")
    


    
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

    if model_name == "ftt" and data_obj.cat_cols:
        model.load_model(poisoned_model_address_toload, model_type="original")
        model.to(device, model_type="original")
    else:
        model.load_model(poisoned_model_address_toload)
        model.to(device)


    attack = Attack(device=device, model=model, data_obj=data_obj, target_label=target_label, mu=mu, beta=beta, lambd=lambd, epsilon=epsilon)

    attack.load_poisoned_dataset()

    poisoned_trainset, poisoned_testset = attack.poisoned_dataset
    poisoned_train_samples, poisoned_test_samples = attack.poisoned_samples
    attack.poisoned_dataset = (poisoned_trainset, poisoned_testset)
    attack.poisoned_samples = (poisoned_train_samples, poisoned_test_samples)


    # Assuming poisoned_trainset and poisoned_train_samples are PyTorch datasets or tensors
    poisoned_indices = []


    # Convert poisoned_train_samples to a set for faster lookup
    poisoned_samples_set = set(tuple(sample.tolist()) for sample, _ in poisoned_train_samples)

    # Iterate over poisoned_trainset to find indices of poisoned samples
    for idx, (sample, _) in enumerate(poisoned_trainset):
        # Convert the sample to a tuple for comparison
        sample_tuple = tuple(sample.tolist())
        
        # Check if the sample is in the poisoned_samples_set
        if sample_tuple in poisoned_samples_set:
            poisoned_indices.append(idx)

    # poisoned_indices now contains the indices of poisoned samples in poisoned_trainset
    # print("Indices of poisoned samples:", poisoned_indices)

    # Step 11: Revert the poisoned dataset to the original categorical features
    FTT = True if attack.model.model_name == "FTTransformer" else False
    if data_obj.cat_cols:
        reverted_poisoned_trainset = attack.data_obj.Revert(attack.poisoned_dataset[0], FTT=FTT)
        reverted_poisoned_testset = attack.data_obj.Revert(attack.poisoned_dataset[1], FTT=FTT)
    else:
        reverted_poisoned_trainset = attack.poisoned_dataset[0]
        reverted_poisoned_testset = attack.poisoned_dataset[1]

    reverted_poisoned_dataset = (reverted_poisoned_trainset, reverted_poisoned_testset)

    # get the clean train and test datasets
    if FTT and data_obj.cat_cols:
        clean_dataset = attack.data_obj.get_normal_datasets_FTT()
    else:
        clean_dataset = attack.data_obj.get_normal_datasets()
    clean_trainset = clean_dataset[0]
    clean_testset = clean_dataset[1]



    # Create a directory for saving the masks 
    masks_dir = Path("./saved_nc_masks")
        
    mask_address = masks_dir / Path(f"{model_name}_{data_obj.dataset_name}_{target_label}_{mu}_{beta}_{lambd}_{epsilon}")

    if not mask_address.exists():
        mask_address.mkdir(parents=True, exist_ok=True)


    # number of features (columns) in the dataset
    num_features = len(data_obj.feature_names)

    # Calculate the minimum and maximum values for each feature in the dataset
    features_min = data_obj.X_encoded.min().values.astype(np.float32)
    features_max = data_obj.X_encoded.max().values.astype(np.float32)

    dataloader = DataLoader(clean_testset, batch_size=32, shuffle=True)


    reverse_trigger.tabular_visualize_label_scan(results_dir=mask_address,
                                                num_features=num_features,
                                                num_classes=data_obj.num_classes,
                                                feature_min=features_min, 
                                                feature_max=features_max, 
                                                model=model, 
                                                dataloader=dataloader,
                                                ftt=FTT)
    

    mad_outlier_detection.analyze_pattern_norm_dist(results_dir=mask_address, num_classes=data_obj.num_classes)

    exit()
    












        
        
    print("==> [Step 4] Fine-Tuning the pruned model on clean data...")
    #Train the model on the poisoned training dataset
    converted = False if FTT and data_obj.cat_cols else True
    attack.train(clean_dataset, converted=converted)
    logging.info("=== Fine-Tuning Completed ===")

    print("==> [Step 5] Testing the pruned model on clean data and poisoned data...")
    asr = attack.test(reverted_poisoned_testset, converted=converted)
    cda = attack.test(clean_testset, converted=converted)
    print("=== Testing Completed ===")


    # creating and checking the path for saving and loading the pruned models.
    pruned_model_path = models_path / Path(f"pruned")
    if not pruned_model_path.exists():
        pruned_model_path.mkdir(parents=True, exist_ok=True)
    pruned_model_address = pruned_model_path / Path(f"{model_name}_{data_obj.dataset_name}_{target_label}_{mu}_{beta}_{lambd}_{epsilon}_pruned_model.pth")

    # Save the poisoned model with Unix timestamp in the filename
    if model_name == "ftt" and data_obj.cat_cols:
        attack.model.save_model(pruned_model_address, model_type="original")
    else:
        attack.model.save_model(pruned_model_address)


