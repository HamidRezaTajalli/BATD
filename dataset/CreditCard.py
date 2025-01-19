from pathlib import Path
import pickle
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


class CreditCard:
    """
    This class is used to load the CreditCard dataset and convert it into a format suitable for training a model.
    """

    def __init__(self, test_size=0.2, random_state=42, batch_size=64):
        self.dataset_name = "credit_card"
        self.num_classes = 2

        # loading the dataset to a Pandas DataFrame
        credit_card_data = pd.read_csv('/home/htajalli/prjs0962/repos/BATD/data/creditcard.csv')

        # separating the features and target
        X = credit_card_data.drop('Class', axis=1)
        y = credit_card_data['Class']


        # Define the categorical and numerical columns
        self.cat_cols = []

        self.num_cols = X.columns.tolist()

        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size

        # Retrieve the feature names from the dataset: in the same order as they appear in the dataset
        self.feature_names = X.columns.tolist()

        self.column_idx = {col: idx for idx, col in enumerate(self.feature_names)}
        self.num_cols_idx = [self.column_idx[col] for col in self.num_cols]
        self.FTT_n_categories = ()

        # Store original dataset for reference
        self.X_original = X.copy()
        self.y = y.copy()

        self.X_encoded = self.X_original.copy()

        # Apply StandardScaler to numerical features to standardize them
        scaler = StandardScaler()
        self.X_encoded[self.num_cols] = scaler.fit_transform(self.X_encoded[self.num_cols])


    

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
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        # X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        # y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
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
#     credit_card = CreditCard()
