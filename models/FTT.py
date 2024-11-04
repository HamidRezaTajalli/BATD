import torch
import torch.nn as nn
from tab_transformer_pytorch import FTTransformer
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
import numpy as np



class FTTModel:
    def __init__(self, data_obj):
        """
        Initializes a TabNet classifier model with customizable hyperparameters.
        
        Parameters:
        n_d (int): Dimension of the decision prediction layer.
        n_a (int): Dimension of the attention embedding.
        n_steps (int): Number of steps in the architecture.
        gamma (float): Relaxation parameter for TabNet architecture.
        n_independent (int): Number of independent Gated Linear Units layers.
        n_shared (int): Number of shared Gated Linear Units layers.
        momentum (float): Momentum for the batch normalization.
        

        """
        self.model_name = "FTTransformer"
        self.data_obj = data_obj
        self.epochs = 65
        self.device = None





        # Initialize the FTTransformer model
        self.model_original = FTTransformer(categories=data_obj.FTT_n_categories,
                                   num_continuous=len(data_obj.num_cols),
                                   dim=128,
                                   depth=6,
                                   heads=8,
                                   dim_head=16,
                                   dim_out=data_obj.num_classes,
                                   num_special_tokens=2,
                                   attn_dropout=0.1,
                                   ff_dropout=0.1,
                                   )

        self.model_converted = FTTransformer(categories=(),
                                   num_continuous=len(data_obj.feature_names),
                                   dim=128,
                                   depth=6,
                                   heads=8,
                                   dim_head=16,
                                   dim_out=data_obj.num_classes,
                                   num_special_tokens=1,
                                   attn_dropout=0.1,
                                   ff_dropout=0.1,
                                   )



    def to(self, device: torch.device, model_type: str = "converted"):
        """
        Moves the model to the specified device.

        Parameters:
        device (torch.device): The device to move the model to.
        model_type (str): The type of model to move to the device.
        """
        if model_type == "converted":
            self.model_converted.to(device)  # Set the device for the FTTransformer model
        elif model_type == "original":
            self.model_original.to(device)  # Set the device for the FTTransformer model    
        else:
            raise ValueError(f"Invalid model type: {model_type}. Please choose 'converted' or 'original'.")
        self.device = device

    

    def get_model(self, model_type: str = "converted"):
        """
        Returns the initialized FTTransformer model for use in training or evaluation.

        Returns:
        model (FTTransformer): The FTTransformer instance.
        """
        if model_type == "converted":
            return self.model_converted
        elif model_type == "original":
            return self.model_original
        else:
            raise ValueError(f"Invalid model type: {model_type}. Please choose 'converted' or 'original'.")

    def fit(self, train_dataset, val_dataset):
        """
        Trains the FTTransformer model using the provided training data.
        
        Parameters:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        
        Returns:
        None
        """
        # Train the model
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device, model_type="original")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model_original.parameters(), lr=3.762989816330166e-05, weight_decay=0.0001239780004929955)

        # 11. Training loop with Early Stopping
        epochs = self.epochs
        best_val_loss = float('inf')
        patience = 20
        trigger_times = 0

        for epoch in range(1, epochs + 1):
            self.model_original.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for X_c, X_n, y_batch in train_loader:
                X_c, X_n, y_batch = X_c.to(device), X_n.to(device), y_batch.to(device)
                

                optimizer.zero_grad()
                outputs = self.model_original(X_c, X_n)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * X_c.size(0)
                
                 
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
            
            avg_train_loss = total_loss / total
            train_accuracy = correct / total
            
            # Validation
            self.model_original.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for X_c, X_n, y_batch in val_loader:
                    X_c, X_n, y_batch = X_c.to(device), X_n.to(device), y_batch.to(device)

                    outputs = self.model_original(X_c, X_n)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_c.size(0)


                    _, predicted = torch.max(outputs, 1)


                    correct_val += (predicted == y_batch).sum().item()
                    total_val += y_batch.size(0)
            
            avg_val_loss = val_loss / total_val
            val_accuracy = correct_val / total_val
            
            print(f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
            
            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                trigger_times = 0
                torch.save(self.model_original.state_dict(), f'best_fttransformer_original_{self.data_obj.dataset_name}.pth')
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("Early stopping!")
                    break

        # 12. Load the best model and evaluate
        self.model_original.load_state_dict(torch.load(f'best_fttransformer_original_{self.data_obj.dataset_name}.pth'))
        self.model_original.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_c, X_n, y_batch in val_loader:
                X_c, X_n = X_c.to(device), X_n.to(device)

                outputs = self.model_original(X_c, X_n)

                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        print("Classification Report:")
        print(classification_report(all_targets, all_preds))

        print("Confusion Matrix:")
        print(confusion_matrix(all_targets, all_preds))





    def fit_converted(self, train_dataset, val_dataset):
        """
        Trains the FTTransformer model using the provided training data.
        
        Parameters:
        
        Returns:
        None
        """
        # Train the model
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device, model_type="converted")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model_converted.parameters(), lr=3.762989816330166e-05, weight_decay=0.0001239780004929955)

        # 11. Training loop with Early Stopping
        epochs = self.epochs
        best_val_loss = float('inf')
        patience = 20
        trigger_times = 0

        for epoch in range(1, epochs + 1):
            self.model_converted.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for X_n, y_batch in train_loader:
                X_c = torch.empty(X_n.shape[0], 0, dtype=torch.long)
                X_c, X_n, y_batch = X_c.to(device), X_n.to(device), y_batch.to(device)
               
                optimizer.zero_grad()
                outputs = self.model_converted(X_c, X_n)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * X_c.size(0)
                
 
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
            
            avg_train_loss = total_loss / total
            train_accuracy = correct / total
            
            # Validation
            self.model_converted.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for X_n, y_batch in val_loader:
                    X_c = torch.empty(X_n.shape[0], 0, dtype=torch.long)
                    X_c, X_n, y_batch = X_c.to(device), X_n.to(device), y_batch.to(device)

                    outputs = self.model_converted(X_c, X_n)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_c.size(0)
                    

                    _, predicted = torch.max(outputs, 1)

                    correct_val += (predicted == y_batch).sum().item()
                    total_val += y_batch.size(0)
            
            avg_val_loss = val_loss / total_val
            val_accuracy = correct_val / total_val
            
            print(f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
            
            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                trigger_times = 0
                torch.save(self.model_converted.state_dict(), f'best_fttransformer_converted_{self.data_obj.dataset_name}.pth')
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("Early stopping!")
                    break

        # 12. Load the best model and evaluate
        self.model_converted.load_state_dict(torch.load(f'best_fttransformer_converted_{self.data_obj.dataset_name}.pth'))
        self.model_converted.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_n, y_batch in val_loader:
                X_c = torch.empty(X_n.shape[0], 0, dtype=torch.long)
                X_c, X_n = X_c.to(device), X_n.to(device)

                outputs = self.model_converted(X_c, X_n)

 
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        print("Classification Report:")
        print(classification_report(all_targets, all_preds))

        print("Confusion Matrix:")
        print(confusion_matrix(all_targets, all_preds))

    
    def predict(self, X_test):
        """
        Makes predictions on the provided test data using the trained TabNet model.
        
        Parameters:
        X_test (array-like): Input features for testing.
        
        Returns:
        accuracy (float): Accuracy of the model on the test data.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device, model_type="original")
        val_loader = DataLoader(X_test, batch_size=1024, shuffle=False)
        self.model_original.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_c, X_n, y_batch in val_loader:
                X_c, X_n = X_c.to(device), X_n.to(device)

                outputs = self.model_original(X_c, X_n)

                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        
        # Convert lists to NumPy arrays for element-wise comparison
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        print("Classification Report:")
        print(classification_report(all_targets, all_preds))

        print("Confusion Matrix:")
        print(confusion_matrix(all_targets, all_preds))
        
        # calculate accuracy
        accuracy = (all_preds == all_targets).mean()
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def predict_converted(self, X_test):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device, model_type="converted")
        val_loader = DataLoader(X_test, batch_size=1024, shuffle=False)
        self.model_converted.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_n, y_batch in val_loader:
                X_c = torch.empty(X_n.shape[0], 0, dtype=torch.long)
                X_c, X_n = X_c.to(device), X_n.to(device)

                outputs = self.model_converted(X_c, X_n)

 
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        # Convert lists to NumPy arrays for element-wise comparison
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        print("Classification Report:")
        print(classification_report(all_targets, all_preds))

        print("Confusion Matrix:")
        print(confusion_matrix(all_targets, all_preds))

        # Calculate accuracy
        accuracy = (all_preds == all_targets).mean()
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy
        
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Return logits for the input features

        X_c = torch.empty(X.shape[0], 0, dtype=torch.long)
        X_c = X_c.to(self.device)
        X_n = X

        proba = self.model_converted(X_c, X_n)
        return proba
        
    

    def save_model(self, filepath, model_type: str = "converted"):
        """
        Saves the trained TabNet model to the specified file path.
        
        Parameters:
        filepath (str): File path to save the model.
        
        Returns:
        None
        """
        if model_type == "converted":
            self.model_converted.save_model(filepath)
        elif model_type == "original":
            self.model_original.save_model(filepath)
        else:
            raise ValueError(f"Invalid model type: {model_type}. Please choose 'converted' or 'original'.")
    
    def load_model(self, filepath, model_type: str = "converted"):
        """
        Loads a pre-trained TabNet model from the specified file path.
        
        Parameters:
        filepath (str): File path to load the model from.
        
        Returns:
        None
        """
        if model_type == "converted":
            self.model_converted.load_model(filepath)
        elif model_type == "original":
            self.model_original.load_model(filepath)
        else:
            raise ValueError(f"Invalid model type: {model_type}. Please choose 'converted' or 'original'.")

    
    def eval(self, model_type: str = "converted"):
        if model_type == "converted":
            self.model_converted.eval()
        elif model_type == "original":
            self.model_original.eval()
        else:
            raise ValueError(f"Invalid model type: {model_type}. Please choose 'converted' or 'original'.")



# Example usage:
# Initialize a TabNet model
# tabnet_model = TabNetModel(input_dim=54, output_dim=7)
# Train the model
# tabnet_model.fit(X_train, y_train, X_valid=X_val, y_valid=y_val)
# Get the trained model
# model = tabnet_model.get_model()



# if __name__ == "__main__":
#     from CovType import CovType
#     data_obj = CovType()
#     model = FTTModel(data_obj)
#     dataset = data_obj.get_normal_datasets_FTT()
#     model.fit(dataset[0], dataset[1])
