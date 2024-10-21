from models.Tabnet import TabNetModel  # Import the TabNet model from tabnet.py
from dataset.CovType import CovType   # Import the CovType dataset handling from covertype.py
import torch

class TrainTestModel:
    def __init__(self, input_dim, output_dim, batch_size=1024, max_epochs=65, patience=65, virtual_batch_size=128):
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
        self.virtual_batch_size = virtual_batch_size
        # Initialize the CovType dataset
        self.dataset = CovType(batch_size=self.batch_size)

        # Load the data
        self.train_set, self.test_set = self.dataset.load_data()
        self.steps_per_epoch = len(self.train_set) // self.batch_size

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the TabNet model with best parameters
        self.model = TabNetModel(
            device=self.device,
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            momentum=0.3,
            mask_type='entmax'
        )

        


    def train(self):
        """
        Trains the TabNet model on the training set.

        Returns:
        None
        """
        # Convert training data to the required format
        X_train, y_train = self._get_dataset_data(self.train_set)


        # Train the model
        self.model.fit(
            X_train=X_train, 
            y_train=y_train, 
            max_epochs=self.max_epochs, 
            patience=self.patience, 
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size
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

        # Calculate accuracy using NumPy
        accuracy = (preds == y_test).mean()
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        return accuracy

    

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
