# importing all the models for this project
import argparse
import csv
from pathlib import Path
import torch




# Use relative imports since we're now in the src package
from src.models.XGBoost import XGBoostModel

# importing all the datasets for this project
from src.dataset.BM import BankMarketing
from src.dataset.ACI import ACI
from src.dataset.HIGGS import HIGGS
from src.dataset.Diabetes import Diabetes
from src.dataset.Eye_Movement import EyeMovement
from src.dataset.KDD99 import KDD99
from src.dataset.CreditCard import CreditCard
from src.dataset.CovType import CovType  # Importing the ConvertCovType class from CovType.py




def clean_train(dataset_name, model_name, method):

    # log the settings 
    print("--------------------------------")
    print('################################')
    print(f"training with dataset: {dataset_name}, model: {model_name}, method: {method}")
    print('################################')
    print("--------------------------------")


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


    data_obj = dataset_dict[dataset_name]()


    dataset = data_obj.get_normal_datasets()


    train_dataset, val_dataset = dataset
    

    
    objective = "multi" if data_obj.num_classes > 2 else "binary"
    model = XGBoostModel(objective=objective)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)



#################################################
    # Training the model 
#################################################

    # Convert training data to the required format
    X_train, y_train = data_obj._get_dataset_data(train_dataset)


    # Train the model
    model.fit(
        X_train=X_train, 
        y_train=y_train
    )


#################################################
    # Extract and print feature importance
#################################################
    
    # Get feature importance from the XGBoost model
    feature_importances = model.model.feature_importances_
    
    # Get feature names from the dataset object
    feature_names = data_obj.feature_names if hasattr(data_obj, 'feature_names') else [f"Feature_{i}" for i in range(len(feature_importances))]
    
    # Get numerical feature indices from the dataset object
    numerical_indices = data_obj.num_cols_idx if hasattr(data_obj, 'num_cols_idx') else list(range(len(feature_importances)))
    
    # Create a list of (feature_index, feature_name, importance_value) tuples for numerical features only
    feature_importance_tuples = [(i, feature_names[i], importance) 
                               for i, importance in enumerate(feature_importances)
                               if i in numerical_indices]
    
    # Sort the numerical features by importance in descending order
    sorted_feature_importance = sorted(feature_importance_tuples, key=lambda x: x[2], reverse=True)
    
    # Get the top 3 most important numerical features
    top_3_features = sorted_feature_importance[:3]
    
    print("\nTop 3 Most Important Numerical Features:")
    for idx, (feature_idx, feature_name, importance) in enumerate(top_3_features, 1):
        print(f"{idx}. {feature_name} (index {feature_idx}): Importance = {importance:.4f}")
    
    # Create a list of the top 3 numerical feature names
    top_3_feature_names = [feature_name for _, feature_name, _ in top_3_features]
    print(f"\nList of top 3 numerical feature names: {top_3_feature_names}")
    
    # Create a list of the top 3 numerical feature indices
    top_3_feature_indices = [feature_idx for feature_idx, _, _ in top_3_features]
    print(f"List of top 3 numerical feature indices: {top_3_feature_indices}")
    
    
#################################################
    # testing the model
#################################################

        
    # Convert test data to the required format
    X_test, y_test = data_obj._get_dataset_data(val_dataset)

            # Make predictions on the test data
    preds = model.predict(X_test)

    # Calculate accuracy using NumPy
    accuracy = (preds == y_test).mean()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    test_accuracy = accuracy * 100




if __name__ == "__main__":

    # create argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    available_datasets = ["aci", "bm", "higgs", "eye_movement", "credit_card", "covtype"]
    available_models = ["xgboost"]
    available_methods = ["ordinal"]

    args.model = "xgboost"
    args.method = "ordinal"

    # check and make sure that the dataset and model are available
    # the dataset and model names are case insensitive, so compare the lowercase versions of the arguments
    if args.dataset.lower() not in available_datasets:
        raise ValueError(f"Dataset {args.dataset} is not available. Please choose from: {available_datasets}")
    if args.model.lower() not in available_models:
        raise ValueError(f"Model {args.model} is not available. Please choose from: {available_models}")
    if args.method.lower() not in available_methods:
        raise ValueError(f"Method {args.method} is not available. Please choose from: {available_methods}")
    

    for dataset in available_datasets:
        clean_train(dataset, args.model, args.method)