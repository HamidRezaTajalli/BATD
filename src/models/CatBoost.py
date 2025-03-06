import os
import torch
import torch.nn as nn
from catboost import CatBoostClassifier

# Define a fallback for the number of estimators
# n_estimators = 1000 if not os.getenv("CI", False) else 20

class CatBoostModel:
    def __init__(self, 
                 objective="multi",
                 depth=8,
                 learning_rate=0.05,
                 random_seed=0,
                 l2_leaf_reg=3.0,
                 subsample=None,
                 rsm=1.0, 
                 random_strength=1.0,
                 bagging_temperature=1.0,
                 border_count=128,
                 boosting_type="Plain",
                 bootstrap_type="Bayesian",
                 thread_count=-1,
                 logging_level="Silent",
                 od_type="Iter",
                 od_wait=40):
        """
        Initializes a CatBoost classifier model with customizable hyperparameters.
        
        Parameters:
        objective (str): 
            - "multi" for multi-class classification (CatBoost will use "MultiClass") 
            - "binary" for binary classification (CatBoost will use "Logloss").
        depth (int): Depth of each tree.
        learning_rate (float): Step size shrinkage used to prevent overfitting.
        n_estimators (int): Number of boosting iterations (trees).
        random_seed (int): Random seed for reproducibility.
        l2_leaf_reg (float): L2 regularization coefficient.
        subsample (float): Fraction of objects to use for training.
        rsm (float): Subsample ratio of features (analogous to colsample_bytree).
        random_strength (float): Amount of randomness to use for scoring splits.
        bagging_temperature (float): Controls intensity of Bayesian bagging. 
        border_count (int): The number of splits for numerical features.
        boosting_type (str): "Ordered" or "Plain". "Plain" is usually faster.
        bootstrap_type (str): "Bayesian", "Bernoulli", "MVS", etc.
        thread_count (int): Number of threads used for training. -1 uses all.
        logging_level (str): Log verbosity. E.g., "Silent", "Verbose", "Info".
        od_type (str): Overfitting detector type ("IncToDec", "Iter").
        od_wait (int): Number of iterations to continue training after the iteration with the best result.
        """
        self.model_name = "CatBoost"
        

        self.objective = objective
        self.depth = depth
        self.learning_rate = learning_rate
        self.n_iterations = 1500
        self.random_seed = random_seed
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.rsm = rsm
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.border_count = border_count
        self.boosting_type = boosting_type
        self.bootstrap_type = bootstrap_type
        self.thread_count = thread_count
        self.logging_level = logging_level
        self.od_type = od_type
        self.od_wait = od_wait

        # Initialize the CatBoost model
        if objective == "binary":
            self.model = CatBoostClassifier(
            loss_function='Logloss',
            depth=self.depth,
            learning_rate=self.learning_rate,
            iterations=self.n_iterations,
            random_seed=self.random_seed,
            l2_leaf_reg=self.l2_leaf_reg,
            subsample=self.subsample,
            rsm=self.rsm,
            random_strength=self.random_strength,
            bagging_temperature=self.bagging_temperature,
            border_count=self.border_count,
            boosting_type=self.boosting_type,
            bootstrap_type=self.bootstrap_type,
            thread_count=self.thread_count,
            logging_level=self.logging_level,
            od_type=self.od_type,
            od_wait=self.od_wait,
            verbose=100
            )
        elif objective == "multi":
            self.model = CatBoostClassifier(
            iterations=10000,              # More iterations for complex data
            learning_rate=0.01,           # Slower learning rate for better convergence
            depth=10,                     # Deeper trees for capturing complex patterns
            l2_leaf_reg=3,                # L2 regularization to prevent overfitting
            border_count=128,             # More precise splits
            loss_function='MultiClass',
            eval_metric='Accuracy',
            task_type='GPU',              # Change to 'GPU' if available
            bootstrap_type='Bayesian',    # Advanced bootstrapping
            bagging_temperature=1,
            verbose=100,
            early_stopping_rounds=50      # Stop if no improvement for 50 rounds
            )
        else:
            raise ValueError(f"Objective '{objective}' not supported. Use 'multi' or 'binary'.")
        
        
        

    def to(self, device: torch.device):
        """
        Moves the model to the specified device.
        (CatBoost training is CPU/GPU-based internally, so typically
        this method can be left as a no-op or used for custom logic.)
        """
        pass

    def get_model(self):
        """
        Returns the initialized CatBoost classifier for use in training or evaluation.
        
        Returns:
        model (CatBoostClassifier): The CatBoost classifier instance.
        """
        return self.model

    def fit(self, 
            X_train, 
            y_train, 
            X_valid=None, 
            y_valid=None):
        """
        Trains the CatBoost model using the provided training data.
        
        Parameters:
        X_train (array-like): Input features for training.
        y_train (array-like): Labels for training.
        X_valid (array-like): Input features for validation (optional).
        y_valid (array-like): Labels for validation (optional).
        early_stopping (int): Number of rounds with no improvement before stopping training early.
        verbose (int): Verbosity level for training (0 = silent, higher = more info).
        
        Returns:
        None
        """
        # If we have validation data, pass it into the 'eval_set'
        eval_set = None
        if X_valid is not None and y_valid is not None:
            eval_set = (X_valid, y_valid)

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set
        )

    def predict(self, X_test):
        """
        Makes predictions on the provided test data using the trained CatBoost model.
        
        Parameters:
        X_test (array-like): Input features for testing.
        
        Returns:
        preds (array-like): Predicted labels for the test data.
        """
        return self.model.predict(X_test)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns prediction probabilities (logits) for the input features as a PyTorch tensor.
        
        Parameters:
        X (torch.Tensor): Input features in a PyTorch tensor.
        
        Returns:
        torch.Tensor: Probability estimates from the CatBoost model.
        """
        # Convert the PyTorch tensor to numpy if necessary
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        proba = self.model.predict_proba(X)
        return torch.tensor(proba)

    def save_model(self, filepath):
        """
        Saves the trained CatBoost model to the specified file path.
        
        Parameters:
        filepath (str): File path to save the model.
        
        Returns:
        None
        """
        self.model.save_model(filepath)

    def load_model(self, filepath):
        """
        Loads a pre-trained CatBoost model from the specified file path.
        
        Parameters:
        filepath (str): File path to load the model from.
        
        Returns:
        None
        """
        self.model.load_model(filepath)

    def eval(self):
        """
        Placeholder for evaluation mode setting.
        Not strictly required for CatBoost models.
        """
        pass


# Example usage (comment out if you prefer to keep this as a library):

# if __name__ == "__main__":
#     cat_model = CatBoostModel(objective="binary")
#     from ACI import ACI
#     data_obj = ACI()
#     train_dataset, test_dataset = data_obj.get_normal_datasets()
#     X_train, y_train = data_obj._get_dataset_data(train_dataset)
#     cat_model.fit(X_train, y_train)
#     X_test, y_test = data_obj._get_dataset_data(test_dataset)
#     preds = cat_model.predict(X_test)
#     accuracy = (preds == y_test).mean()
#     print(f"Accuracy: {accuracy * 100:.2f}%")