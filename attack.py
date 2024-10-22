# attack.py

from pathlib import Path
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
import torch
from dataset.CovType import CovType  # Importing the ConvertCovType class from CovType.py
from models.Tabnet import TabNetModel             # Importing the TabNet model from TabNet.py
import joblib                        # For saving and loading models
import os
from torch.utils.data import TensorDataset, DataLoader

# Import time module for Unix timestamp
import time

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
    
    def __init__(self, device, model, data_obj, target_label=1, mu=0.2):
        """
        Initializes the class for performing a backdoor attack on a model.

        Parameters:
        model (TabNetModel): The TabNet model instance.
        data_obj (CovType): The CovType dataset instance.
        target_label (int): The target label for the backdoor attack.
        mu (float, optional): Fraction of non-target samples to select based on confidence scores. Defaults to 0.2.
        """
                


        # Initialize the dataset
        self.data_obj = data_obj
        self.converted_dataset = self.data_obj.get_converted_dataset()

        # Device
        self.device = device

        # Initialize the model
        self.model = model

        self.target_label = target_label

        # Initialize the D_non_target attribute to store non-target samples
        self.D_non_target = None

        # Initialize the D_picked attribute to store the picked samples
        self.D_picked = None

        # set the mu parameter
        self.mu = mu
        
        logging.info(f"Attack initialized with target label: {self.target_label}")
        
    
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
    

    def select_non_target_samples(self) -> TensorDataset:
        """
        Selects non-target samples from the training dataset by excluding samples with the target label.
        
        Constructs the subset \( D_{\text{non-target}} \) which contains all samples not labeled as the target.
        
        Returns:
            D_non_target (TensorDataset): The subset of the training dataset excluding target samples.
        """
        logging.info("Selecting non-target samples from the training dataset...")
        
        # Extract training data
        X_train, y_train = self.data_obj._get_dataset_data(self.converted_dataset[0])
        
        # Convert to NumPy arrays for processing
        X_train_np = X_train
        y_train_np = y_train
        
        # Identify indices where the label is not equal to the target label
        non_target_indices = np.where(y_train_np != self.target_label)[0]
        
        # Select non-target samples based on identified indices
        X_non_target = X_train_np[non_target_indices]
        y_non_target = y_train_np[non_target_indices]
        
        logging.info(f"Selected {len(X_non_target)} non-target samples out of {len(X_train_np)} total training samples.")
        
        # Convert selected non-target samples to PyTorch tensors
        X_non_target_tensor = torch.tensor(X_non_target, dtype=torch.float32)
        y_non_target_tensor = torch.tensor(y_non_target, dtype=torch.long)
        
        # Create a TensorDataset for the non-target samples
        self.D_non_target = TensorDataset(X_non_target_tensor, y_non_target_tensor)
    
        return self.D_non_target
    

    def confidence_based_sample_ranking(self, batch_size=64) -> TensorDataset:
        """
        Performs confidence-based sample ranking to select the top mu fraction of non-target samples
        based on the model's confidence in predicting the target class.

        Steps:
            1. Evaluate the model on D_non_target to obtain softmax confidence scores for the target class.
            2. Pair each input with its confidence score to form D_conf.
            3. Sort D_conf in descending order based on confidence scores.
            4. Select the top mu fraction of samples to create D_picked.

        Args:
            batch_size (int, optional): Number of samples per batch for evaluation. Defaults to 64.
        
        Returns:
            D_picked (TensorDataset): The subset of non-target samples selected based on confidence scores.
        """
        if self.D_non_target is None:
            raise ValueError("D_non_target is not initialized. Please run select_non_target_samples() first.")
        
        logging.info("Starting confidence-based sample ranking...")
        
        # Create a DataLoader for D_non_target
        non_target_loader = DataLoader(self.D_non_target, batch_size=batch_size, shuffle=False)
        
        # Initialize lists to store confidence scores and corresponding indices
        confidence_scores = []
        sample_indices = []
        
        # Disable gradient computation for efficiency
        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(non_target_loader):
                # Move data to the appropriate device
                X_batch = X_batch.to(self.device)
                
                # Forward pass through the model to get logits
                logits = self.model.forward(X_batch)
                
                # Apply softmax to obtain probabilities
                probabilities = torch.softmax(logits, dim=1)
                
                # Extract the probability of the target class for each sample in the batch
                # Assuming target_label is zero-indexed
                s_batch = probabilities[:, self.target_label]
                
                # Move probabilities to CPU and convert to NumPy
                s_batch_np = s_batch.cpu().numpy()
                
                # Calculate the absolute sample indices
                # Assuming DataLoader iterates in order without shuffling
                start_idx = batch_idx * batch_size
                end_idx = start_idx + len(s_batch_np)
                batch_indices = np.arange(start_idx, end_idx)
                
                # Append to the lists
                confidence_scores.extend(s_batch_np)
                sample_indices.extend(batch_indices)
        
        # Convert lists to NumPy arrays for efficient processing
        confidence_scores = np.array(confidence_scores)
        sample_indices = np.array(sample_indices)
        
        # Pair each sample index with its confidence score
        D_conf = list(zip(sample_indices, confidence_scores))
        
        # Sort D_conf in descending order based on confidence scores
        D_conf_sorted = sorted(D_conf, key=lambda x: x[1], reverse=True)
        
        # Determine the number of samples to pick based on mu
        total_non_target = len(D_conf_sorted)
        top_mu_count = int(self.mu * total_non_target)
        logging.info(f"Selecting top {self.mu*100:.1f}% ({top_mu_count}) samples based on confidence scores.")
        
        # Select the top mu fraction of samples
        D_picked_indices = [idx for idx, score in D_conf_sorted[:top_mu_count]]
        
        # Retrieve the corresponding samples from D_non_target
        X_picked = []
        y_picked = []
        for idx in D_picked_indices:
            X_picked.append(self.D_non_target[idx][0].cpu())
            y_picked.append(self.D_non_target[idx][1].cpu())
        
        # Convert lists to tensors
        X_picked_tensor = torch.stack(X_picked)
        y_picked_tensor = torch.tensor(y_picked, dtype=torch.long)
        
        # Create a TensorDataset for the picked samples
        self.D_picked = TensorDataset(X_picked_tensor, y_picked_tensor)
        
        logging.info(f"Selected {len(self.D_picked)} samples for further processing.")
        
        return self.D_picked


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
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            momentum=0.3,
            mask_type='entmax'
        )
        
    model.to(device)


    attack = Attack(device=device, model=model, data_obj=data_obj)

    # load the trained model
    attack.model.load_model("./saved_models/tabnet_1729536738.zip")
    attack.model.to(device)
    

    # attack.train()
    attack.test()

    # Get current Unix timestamp
    unix_timestamp = int(time.time())

    # # Save the model with Unix timestamp in the filename
    # attack.model.save_model(f"./saved_models/tabnet_{unix_timestamp}")
    
    logging.info("=== Initial Model Training Completed ===")


    D_non_target = attack.select_non_target_samples()
    D_picked = attack.confidence_based_sample_ranking()




# Execute the main function when the script is run directly
if __name__ == "__main__":
    main()


# TODO: test the two new methods: confidence_based_sample_ranking