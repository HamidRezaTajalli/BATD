 import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier

class TabNetModel:
    def __init__(self, input_dim, output_dim, n_d=64, n_a=64, n_steps=5, gamma=1.5,
                 n_independent=2, n_shared=2, lambda_sparse=1e-3, momentum=0.3,
                 clip_value=2.0, optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-3),
                 scheduler_fn=torch.optim.lr_scheduler.OneCycleLR, scheduler_params=None, verbose=1):
        """
        Initializes a TabNet classifier model with customizable hyperparameters.
        
        Parameters:
        input_dim (int): Number of input features.
        output_dim (int): Number of classes for classification (output size).
        n_d (int): Dimension of the decision prediction layer.
        n_a (int): Dimension of the attention embedding.
        n_steps (int): Number of steps in the architecture.
        gamma (float): Relaxation parameter for TabNet architecture.
        n_independent (int): Number of independent Gated Linear Units layers.
        n_shared (int): Number of shared Gated Linear Units layers.
        lambda_sparse (float): Coefficient for feature sparsity loss.
        momentum (float): Momentum for the batch normalization.
        clip_value (float): Value at which gradients will be clipped.
        optimizer_fn (function): Optimizer function.
        optimizer_params (dict): Parameters for the optimizer.
        scheduler_fn (function): Learning rate scheduler function.
        scheduler_params (dict): Parameters for the scheduler.
        verbose (int): Verbosity level (0 = silent, 1 = verbose).
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.momentum = momentum
        self.clip_value = clip_value
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params
        self.scheduler_fn = scheduler_fn
        self.scheduler_params = scheduler_params
        self.verbose = verbose

        # Initialize the TabNet model
        self.model = TabNetClassifier(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            lambda_sparse=self.lambda_sparse,
            momentum=self.momentum,
            clip_value=self.clip_value,
            optimizer_fn=self.optimizer_fn,
            optimizer_params=self.optimizer_params,
            scheduler_fn=self.scheduler_fn,
            scheduler_params=self.scheduler_params,
            verbose=self.verbose
        )
    
    def get_model(self):
        """
        Returns the initialized TabNet model for use in training or evaluation.
        
        Returns:
        model (TabNetClassifier): The TabNet classifier instance.
        """
        return self.model

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, max_epochs=100, patience=10, batch_size=1024, virtual_batch_size=128):
        """
        Trains the TabNet model using the provided training data.
        
        Parameters:
        X_train (array-like): Input features for training.
        y_train (array-like): Labels for training.
        X_valid (array-like): Input features for validation (optional).
        y_valid (array-like): Labels for validation (optional).
        max_epochs (int): Maximum number of epochs for training.
        patience (int): Number of epochs with no improvement before stopping training early.
        batch_size (int): Batch size for training.
        virtual_batch_size (int): Size of virtual batches for memory-efficient training.
        
        Returns:
        None
        """
        # Train the model
        self.model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)] if X_valid is not None and y_valid is not None else None,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size
        )
    
    def predict(self, X_test):
        """
        Makes predictions on the provided test data using the trained TabNet model.
        
        Parameters:
        X_test (array-like): Input features for testing.
        
        Returns:
        preds (array-like): Predicted labels for the test data.
        """
        return self.model.predict(X_test)

    def save_model(self, filepath):
        """
        Saves the trained TabNet model to the specified file path.
        
        Parameters:
        filepath (str): File path to save the model.
        
        Returns:
        None
        """
        self.model.save_model(filepath)
    
    def load_model(self, filepath):
        """
        Loads a pre-trained TabNet model from the specified file path.
        
        Parameters:
        filepath (str): File path to load the model from.
        
        Returns:
        None
        """
        self.model.load_model(filepath)

# Example usage:
# Initialize a TabNet model
# tabnet_model = TabNetModel(input_dim=54, output_dim=7)
# Train the model
# tabnet_model.fit(X_train, y_train, X_valid=X_val, y_valid=y_val)
# Get the trained model
# model = tabnet_model.get_model()
