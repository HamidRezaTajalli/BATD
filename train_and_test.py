from models.Tabnet import TabNetModel  # Import the TabNet model from tabnet.py
from dataset.CovType import CovType   # Import the CovType dataset handling from covertype.py
import torch

class TrainTestModel:
    def __init__(self, input_dim, output_dim, batch_size=1024, max_epochs=100, patience=10):
        """
        Initializes the class for training and testing the TabNet model on the CovType dataset.

        Parameters:
        input_dim (int): Number of input features.
        output_dim (int): Number of output classes (for classification).
        batch_size (int): Batch size for training.
        max_epochs (int): Maximum number of epochs for training.
        patience (int): Number of epochs with no improvement before early stopping.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        # Initialize the CovType dataset
        self.dataset = CovType(batch_size=self.batch_size)

        # Load the data
        self.train_set, self.test_set = self.dataset.load_data()
        self.steps_per_epoch = len(self.train_set) // self.batch_size

        # Initialize the TabNet model with best parameters
        self.model = TabNetModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            n_d=64,  # Dimension of the decision prediction layer
            n_a=64,  # Dimension of the attention embedding
            n_steps=5,  # Number of steps in the architecture
            gamma=1.5,  # Relaxation parameter for TabNet architecture
            lambda_sparse=1e-3,  # Coefficient for feature sparsity loss
            optimizer_fn=torch.optim.Adam,  # Optimizer function
            optimizer_params=dict(lr=2e-3),  # Learning rate
            scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,  # Learning rate scheduler function
            scheduler_params={
                'max_lr': 1e-2,
                'epochs': self.max_epochs,
                'steps_per_epoch': self.steps_per_epoch,
                'pct_start': 0.3,
                'anneal_strategy': 'cos',
                'cycle_momentum': True,
                'base_momentum': 0.85,
                'max_momentum': 0.95,
                'div_factor': 25.0,
                'final_div_factor': 1e4,
            },
            verbose=1,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            steps_per_epoch=self.steps_per_epoch
        )

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        """
        Trains the TabNet model on the training set.

        Returns:
        None
        """
        # Convert training data to the required format
        X_train, y_train = self._get_dataset_data(self.train_set)
        # X_train, y_train = X_train.to(self.device), y_train.to(self.device)


        # Train the model
        self.model.fit(
            X_train=X_train, 
            y_train=y_train, 
            max_epochs=self.max_epochs, 
            patience=self.patience, 
            batch_size=self.batch_size
        )

    def test(self):
        """
        Tests the trained TabNet model on the test set.

        Returns:
        accuracy (float): Accuracy of the model on the test set.
        """
        # Convert test data to the required format
        X_test, y_test = self._get_dataset_data(self.test_set)

        # Make predictions on the test data
        preds = self.model.predict(X_test)

        # Calculate accuracy
        accuracy = (preds == y_test).float().mean().item()
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        return accuracy

    def _get_dataset_data(self, dataset):
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

# Example usage:
if __name__ == "__main__":
    # Set the input and output dimensions for the Forest Cover Type dataset
    input_dim = 54  # Number of features
    output_dim = 7  # Number of forest cover types

    # Initialize the training and testing class
    trainer = TrainTestModel(input_dim=input_dim, output_dim=output_dim)

    # Train the model
    print("Training the model...")
    trainer.train()

    # Test the model
    print("Testing the model...")
    trainer.test()
