# attack.py

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
import torch
from dataset.CovType import CovType  # Importing the ConvertCovType class from CovType.py
from models.Tabnet import TabNetModel             # Importing the TabNet model from TabNet.py
import joblib                        # For saving and loading models
import os

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
# Attack Class Definition
# ============================================

class Attack:
    """
    The Attack class encapsulates the steps required to perform a backdoor attack on a neural network model.
    
    Attributes:
    """
    
    def __init__(self, model, data_obj):
        """
        Initializes the class for performing a backdoor attack on a model.

        Parameters:
        model (TabNetModel): The TabNet model instance.
        data_obj (CovType): The CovType dataset instance.
        """
                


        # Initialize the dataset
        self.data_obj = data_obj
        self.converted_dataset = self.data_obj.get_converted_dataset()

        # Load the data
        self.train_set, self.test_set = self.dataset.load_data()
        self.steps_per_epoch = len(self.train_set) // self.batch_size

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the model
        self.model = model
        
    
    
    def train(self):
        """
        Trains the TabNet model on the training set.

        Returns:
        None
        """
        # Convert training data to the required format
        X_train, y_train = self.data_obj._get_dataset_data(self.converted_dataset[0])


        # Train the model
        self.model.fit(
            X_train=X_train, 
            y_train=y_train
        )
    
    def test(self):
        """
        Tests the trained TabNet model on the test set.

        Returns:
        accuracy (float): Accuracy of the model on the test set.
        """
        # Convert test data to the required format
        X_test, y_test = self.data_obj._get_dataset_data(self.converted_dataset[1])

        # Make predictions on the test data
        preds = self.model.predict(X_test)

        # Calculate accuracy using NumPy
        accuracy = (preds == y_test).mean()
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        return accuracy
    
    def save_model(self, filepath='trained_model.joblib'):
        """
        Saves the trained model to disk for future use.
        
        Args:
            filepath (str, optional): Path to save the model file. Defaults to 'trained_model.joblib'.
        """
        logging.info(f"Saving the trained model to {filepath}...")
        
        # Serialize and save the model using joblib
        joblib.dump(self.model, filepath)
        
        logging.info("Model saved successfully.")
    
    def load_model(self, filepath='trained_model.joblib'):
        """
        Loads a trained model from disk.
        
        Args:
            filepath (str, optional): Path to the model file. Defaults to 'trained_model.joblib'.
        
        Raises:
            FileNotFoundError: If the specified model file does not exist.
        """
        logging.info(f"Loading the trained model from {filepath}...")
        
        # Check if the model file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The model file '{filepath}' does not exist.")
        
        # Load the model using joblib
        self.model = joblib.load(filepath)
        
        logging.info("Model loaded successfully.")

# ============================================
# Main Execution Block
# ============================================

def main():
    """
    The main function orchestrates the attack setup by executing the initial model training steps.
    """
    logging.info("=== Starting Initial Model Training ===")
    
    # Step 1: Initialize the dataset converter
    data_obj = CovType()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Step 2: Initialize the TabNet model
    model = TabNetModel(
            device=device,
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            momentum=0.3,
            mask_type='entmax'
        )

    attack = Attack(model=model, data_obj=data_obj)

    attack.train()
    attack.test()
    
    logging.info("=== Initial Model Training Completed ===")

# Execute the main function when the script is run directly
if __name__ == "__main__":
    main()
