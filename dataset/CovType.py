import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.datasets import fetch_covtype
import torch
from torch.utils.data import DataLoader, TensorDataset

class CovType:
    def __init__(self, test_size=0.2, random_state=42, batch_size=64):
        """
        Initializes the CovType class for loading and processing the CoverType dataset.
        
        Parameters:
        test_size (float): The proportion of the data to use for the test set.
        ## val_size (float): The proportion of the data to use for the validation set.
        random_state (int): A random seed for data splitting.
        batch_size (int): Batch size for the DataLoader.
        """
        self.test_size = test_size
        # self.val_size = val_size
        self.random_state = random_state
        self.batch_size = batch_size

        # Define the categorical and numerical columns
        self.cat_cols = [
            "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3",
            "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4",
            "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9",
            "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14",
            "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19",
            "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24",
            "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29",
            "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34",
            "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39",
            "Soil_Type40"
        ]
        
        self.num_cols = [
            "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
            "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points"
        ]

    def load_data(self):
        """
        Loads the Forest Cover Type dataset using sklearn's fetch_covtype.
        Converts categorical features using OrdinalEncoder and returns the dataset
        split into train, validation, and test sets.
        
        Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
        """
        # Load the CoverType dataset from sklearn
        data = fetch_covtype(as_frame=True)  # Load data as pandas DataFrame
        X = data['data']  # Features
        y = data['target']  # Target (Forest cover type classification)
        
        # Convert categorical columns using OrdinalEncoder
        ordinal_encoder = OrdinalEncoder()
        X[self.cat_cols] = ordinal_encoder.fit_transform(X[self.cat_cols])
        
        # Split the data into train and temporary sets (temporary set will be further split into validation and test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)
        
        # Further split the temporary set into validation and test sets
        # val_size_adjusted = self.val_size / (1 - self.test_size)  # Adjust validation size based on remaining data
        # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state, stratify=y_temp)
        
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
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Return the Datasets for training and test sets
        return train_dataset, test_dataset

# Example usage:
# covtype = CovType()
# train_loader, val_loader, test_loader = covtype.load_data()
