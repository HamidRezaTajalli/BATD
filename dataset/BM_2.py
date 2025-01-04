from pathlib import Path
import pickle
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml


class BankMarketing:
    """
    This class is used to load the Bank Marketing dataset and convert it into a format suitable for training a model.
    """

    def __init__(self, test_size=0.2, random_state=42, batch_size=64):

        self.dataset_name = "bank_marketing"
        self.num_classes = 2

        # Load the Bank Marketing dataset from sklearn
        X, y = fetch_openml(data_id=44126, as_frame=True, return_X_y=True)

        y = y.astype(int)

        # Subtract 1 to make classes start at 0
        y = y - 1


        # Define the categorical and numerical columns
        self.cat_cols = []

        self.num_cols = ['V1', 'V6', 'V10', 'V12', 'V13', 'V14', 'V15']

        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size

        # Retrieve the feature names from the dataset: in the same order as they appear in the dataset
        self.feature_names = X.columns.tolist()

        self.column_idx = {col: idx for idx, col in enumerate(self.feature_names)}
        self.cat_cols_idx = [self.column_idx[col] for col in self.cat_cols]
        self.num_cols_idx = [self.column_idx[col] for col in self.num_cols]

        # Store original dataset for reference
        self.X_original = X.copy()
        self.y = y.copy()


        # Convert categorical columns using OrdinalEncoder
        # This transforms categorical string labels into integer encodings
        # Since, Bank Marketing dataset has no categorical columns, this is not used
        # ordinal_encoder = OrdinalEncoder()
        self.X_encoded = X.copy()

        # For training the FTT model, I need to know the number of unique categories in each categorical feature as a tuple
        # Since, Bank Marketing dataset has no categorical columns, this is not used
        self.FTT_n_categories = None

        # Apply StandardScaler to numerical features to standardize them
        scaler = StandardScaler()
        self.X_encoded[self.num_cols] = scaler.fit_transform(self.X_encoded[self.num_cols])

        # Initialize a dictionary to store the primary mappings for each categorical feature
        # Since, Bank Marketing dataset has no categorical columns, this is not used
        self.primary_mappings = None

        # Initialize a dictionary to store the adaptive Delta r for each categorical feature
        # Since, Bank Marketing dataset has no categorical columns, this is not used
        self.delta_r_values = None

        # Initialize a dictionary to store the hierarchical mappings for each categorical feature
        # Since, Bank Marketing dataset has no categorical columns, this is not used
        self.hierarchical_mappings = None
        
        # Initialize a dictionary to store the lookup tables for reverse mapping
        # Since, Bank Marketing dataset has no categorical columns, this is not used
        self.lookup_tables = None
        

    def get_normal_datasets(self, dataloader=False, batch_size=None, test_size=None, random_state=None):

        if test_size is None:
            test_size = self.test_size
        if random_state is None:
            random_state = self.random_state
        if batch_size is None:
            batch_size = self.batch_size

        # Split the data into train and temporary sets (temporary set will be further split into validation and test)
        X_train, X_test, y_train, y_test = train_test_split(self.X_encoded, self.y, test_size=test_size, random_state=random_state, stratify=self.y)
        
        # Further split the temporary set into validation and test sets
        # val_size_adjusted = self.val_size / (1 - test_size)  # Adjust validation size based on remaining data
        # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp)
        
        # Convert the data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float64)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        # X_val_tensor = torch.tensor(X_val.values, dtype=torch.float64)
        # y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float64)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
        
        # Create TensorDatasets for each split
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        # val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create DataLoader for each split
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Return the Datasets for training and test sets
        if dataloader:
            return train_loader, test_loader
        else:
            return train_dataset, test_dataset


    
    def _get_dataset_data(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts features and labels from the TensorDataset and converts them into appropriate numpy arrays.

        Parameters:
        dataset (TensorDataset): PyTorch TensorDataset object containing the data.

        Returns:
        X (numpy array): Features from the TensorDataset.
        y (numpy array): Labels from the TensorDataset.
        """
        X_tensor, y_tensor = dataset.tensors

        X = X_tensor.cpu().numpy()
        y = y_tensor.cpu().numpy()


        return X, y



# if __name__ == "__main__":
#     dataset = BankMarketing()