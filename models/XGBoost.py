import os
import torch
import torch.nn as nn
import xgboost as xgb



n_estimators = 1000 if not os.getenv("CI", False) else 20

class XGBClassifier:
    def __init__(self, 
        objective="multi",
        max_depth=8,
        learning_rate=0.1,
        n_estimators=n_estimators,
        verbosity=0,
        silent=None,
        booster='gbtree',
        n_jobs=-1,
        nthread=None,
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=0.7,
        colsample_bytree=1,
        colsample_bylevel=1,
        colsample_bynode=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        random_state=0,
        seed=None):
        """
        Initializes a XGBoost classifier model with customizable hyperparameters.
        
        Parameters:
        max_depth (int): Maximum depth of a tree.
        learning_rate (float): Learning rate for the model.
        n_estimators (int): Number of trees in the model.
        verbosity (int): Verbosity level for the model.
        silent (bool): Whether to suppress output.
        objective (str): Objective function for the model.
        booster (str): Booster type for the model.
        n_jobs (int): Number of jobs to run in parallel.
        nthread (int): Number of threads to use.
        gamma (float): Minimum loss reduction required to make a split.
        min_child_weight (int): Minimum sum of instance weight (hessian) needed in a child.
        max_delta_step (float): Maximum delta step we allow each tree.
        subsample (float): Fraction of the training data to use for fitting the trees.
        colsample_bytree (float): Fraction of the features to use for fitting the trees.
        colsample_bylevel (float): Fraction of the features to use for fitting the trees at each level.
        colsample_bynode (float): Fraction of the features to use for fitting the trees at each node.
        reg_alpha (float): L1 regularization term on weights.
        reg_lambda (float): L2 regularization term on weights.
        scale_pos_weight (float): Balancing of positive and negative weights.
        base_score (float): The initial prediction score of all instances, global bias.
        random_state (int): Random seed for reproducibility.
        seed (int): Random seed for reproducibility.
        """

        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.verbosity = verbosity
        self.silent = silent
        if objective == 'multi':
            self.objective = "multi:softmax"
        elif objective == 'binary':
            self.objective = "binary:logistic"
        else:
            raise ValueError(f"Objective {objective} not supported")
        
        self.booster = booster
        self.n_jobs = n_jobs
        self.nthread = nthread
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.random_state = random_state
        self.seed = seed



        # Initialize the XGBoost model
        self.model = xgb.XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            verbosity=self.verbosity,
            silent=self.silent,
            objective=self.objective,
            booster=self.booster,
            n_jobs=-1,
            nthread=None,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bynode=self.colsample_bynode,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            base_score=self.base_score,
            random_state=self.random_state,
            seed=self.seed
        )


    def to(self, device: torch.device):
        """
        Moves the model to the specified device.

        Parameters:
        device (torch.device): The device to move the model to.
        """
        self.model.device = 'cuda' if device.type == 'cuda' else 'cpu'  # Set the device for the TabNet model

    

    def get_model(self):
        """
        Returns the initialized TabNet model for use in training or evaluation.
        
        Returns:
        model (TabNetClassifier): The TabNet classifier instance.
        """
        return self.model

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, early_stopping_rounds=40, verbose=10):
        """
        Trains the TabNet model using the provided training data.
        
        Parameters:
        X_train (array-like): Input features for training.
        y_train (array-like): Labels for training.
        X_valid (array-like): Input features for validation (optional).
        y_valid (array-like): Labels for validation (optional).
        early_stopping_rounds (int): Number of rounds with no improvement before stopping training early.
        verbose (int): Verbosity level for training.
        
        Returns:
        None
        """
        # Train the model
        self.model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)] if X_valid is not None and y_valid is not None else None,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose
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
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Return logits for the input features
        # Convert numpy array to PyTorch tensor
        proba = self.model.predict_proba(X)
        return torch.tensor(proba)
    

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

    
    def eval(self):
        """
        Placeholder for evaluation mode setting.
        Not required for XGBoost models.
        """
        pass



# Example usage:
# Initialize a TabNet model
# tabnet_model = TabNetModel(input_dim=54, output_dim=7)
# Train the model
# tabnet_model.fit(X_train, y_train, X_valid=X_val, y_valid=y_val)
# Get the trained model
# model = tabnet_model.get_model()
