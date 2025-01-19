# importing all the models for this project
import argparse
import csv
from pathlib import Path
import torch
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




def clean_train(dataset_name, model_name, method):

    if method in ['ohe', 'converted']:
        if dataset_name in ['credit_card', 'diabetes', 'higgs']:
            exit()

    # log the settings 
    print("--------------------------------")
    print('################################')
    print(f"training with dataset: {dataset_name}, model: {model_name}, method: {method}")
    print('################################')
    print("--------------------------------")

    # SAINT and FTT need to be trained with is_numerical value set correctly depending on chosen dataset and method
    is_numerical = False
    if dataset_name in ['credit_card', 'diabetes', 'higgs']:
        is_numerical = True
    if method == 'converted' or method == 'ohe':
        is_numerical = True



    models_path = Path("./clean_saved_models")
    if method == "ohe":
        models_path = models_path / Path("ohe")
    elif method == "ordinal":
        models_path = models_path / Path("ordinal")
    elif method == "converted":
        models_path = models_path / Path("converted")
    
    if not models_path.exists():
        models_path.mkdir(parents=True, exist_ok=True)
    
    models_address = models_path / Path(f"{model_name}_{dataset_name}.pth")


    # create the experiment results directory
    results_path = Path("./clean_results")
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    csv_file_address = results_path / Path(f"{dataset_name}.csv")
    if not csv_file_address.exists():
        csv_file_address.touch()
        
        csv_header = ['EXP_NUM', 'METHOD', 'DATASET', 'MODEL', 'BA']
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

    data_obj = dataset_dict[dataset_name]()

    if method == "ohe":
        dataset = data_obj.get_normal_datasets_ohe()
    elif method == "ordinal":
        if model_name == "ftt" and dataset_name not in ["higgs", "diabetes", "credit_card"]:
            dataset = data_obj.get_normal_datasets_FTT()
        else:
            dataset = data_obj.get_normal_datasets()
    elif method == "converted":
        dataset = data_obj.get_converted_dataset()

    train_dataset, val_dataset = dataset
    

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
        model = SAINTModel(data_obj=data_obj, is_numerical=is_numerical)
    elif model_name == "catboost" or model_name == "xgboost":
        objective = "multi" if data_obj.num_classes > 2 else "binary"
        model = model_dict[model_name](objective=objective)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)



#################################################
    # Training the model 
#################################################

    

    if model.model_name == "TabNet" or model.model_name == "XGBoost" or model.model_name == "CatBoost":

        # Convert training data to the required format
        X_train, y_train = data_obj._get_dataset_data(train_dataset)


        # Train the model
        model.fit(
            X_train=X_train, 
            y_train=y_train
        )
    elif model.model_name == "SAINT":
        X_train, y_train = data_obj._get_dataset_data(train_dataset)

        X_val, y_val = data_obj._get_dataset_data(val_dataset)
        
        model.fit(
            X_train=X_train, 
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )
    elif model.model_name == "FTTransformer":
        if is_numerical:
            model.fit_converted(train_dataset, val_dataset)
        else:
            model.fit(train_dataset, val_dataset)
    else:
        raise ValueError(f"Model {model.model_name} not supported.")


    
#################################################
    # testing the model
#################################################

        
    if model.model_name == "TabNet" or model.model_name == "XGBoost" or model.model_name == "CatBoost":
        # Convert test data to the required format
        X_test, y_test = data_obj._get_dataset_data(val_dataset)

                # Make predictions on the test data
        preds = model.predict(X_test)

        # Calculate accuracy using NumPy
        accuracy = (preds == y_test).mean()
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        test_accuracy = accuracy * 100

    elif model.model_name == "FTTransformer":
        if is_numerical:
            accuracy = model.predict_converted(val_dataset)
        else:
            accuracy = model.predict(val_dataset)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        test_accuracy = accuracy * 100

    elif model.model_name == "SAINT":
        X_test, y_test = data_obj._get_dataset_data(val_dataset)
        accuracy = model.predict(X_test, y_test)
        print(f"Test Accuracy: {accuracy}")
        test_accuracy = accuracy

    else:
        raise ValueError(f"Model {model.model_name} not supported.")
    

    # save the model
    model.save_model(models_address)

    # save the results in csv file
    with open(csv_file_address, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([0, method, dataset_name, model_name, test_accuracy])




if __name__ == "__main__":

    # create argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()

    available_datasets = ["aci", "bm", "higgs", "diabetes", "eye_movement", "kdd99", "credit_card", "covtype"]
    available_models = ["ftt", "tabnet", "saint", "catboost", "xgboost"]
    available_methods = ["ohe", "ordinal", "converted"]

    # check and make sure that the dataset and model are available
    # the dataset and model names are case insensitive, so compare the lowercase versions of the arguments
    if args.dataset.lower() not in available_datasets:
        raise ValueError(f"Dataset {args.dataset} is not available. Please choose from: {available_datasets}")
    if args.model.lower() not in available_models:
        raise ValueError(f"Model {args.model} is not available. Please choose from: {available_models}")
    if args.method.lower() not in available_methods:
        raise ValueError(f"Method {args.method} is not available. Please choose from: {available_methods}")


    clean_train(args.dataset, args.model, args.method)