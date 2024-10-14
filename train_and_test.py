from tabnet import TabNetModel  # Import the TabNet model from tabnet.py
from covertype import CovType   # Import the CovType dataset handling from covertype.py
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
        self.train_loader, self.val_loader, self.test_loader = self.dataset.load_data()

        # Initialize the TabNet model
        self.model = TabNetModel(input_dim=self.input_dim, output_dim=self.output_dim)

    def train(self):
        """
        Trains the TabNet model on the training set and evaluates it on the validation set.

        Returns:
        None
        """
        # Convert training and validation data to the required format
        X_train, y_train = self._get_loader_data(self.train_loader)
        X_val, y_val = self._get_loader_data(self.val_loader)

        # Train the model
        self.model.fit(
            X_train=X_train, 
            y_train=y_train, 
            X_valid=X_val, 
            y_valid=y_val, 
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
        X_test, y_test = self._get_loader_data(self.test_loader)

        # Make predictions on the test data
        preds = self.model.predict(X_test)

        # Calculate accuracy
        accuracy = (preds == y_test).float().mean().item()
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        return accuracy

    def _get_loader_data(self, loader):
        """
        Extracts features and labels from the DataLoader and converts them into appropriate numpy arrays.

        Parameters:
        loader (DataLoader): PyTorch DataLoader object containing the data.

        Returns:
        X (numpy array): Features from the DataLoader.
        y (numpy array): Labels from the DataLoader.
        """
        X_list, y_list = [], []
        for batch in loader:
            X_batch, y_batch = batch
            X_list.append(X_batch)
            y_list.append(y_batch)

        # Concatenate all batches into single tensors
        X = torch.cat(X_list).numpy()
        y = torch.cat(y_list).numpy()

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
