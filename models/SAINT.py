import torch
import torch.nn as nn
from pytorch_widedeep.models import SAINT, WideDeep, Wide
from pytorch_widedeep import Trainer
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.self_supervised_training import ContrastiveDenoisingTrainer
from torch.utils.data import DataLoader, TensorDataset



class SAINTClassifier:

    def __init__(self, column_idx, objective, pred_dim, data_obj, metrics=[Accuracy]):
        """
        Initializes a SAINT model with customizable hyperparameters.
        
        Parameters:
        column_idx (dict): Dictionary mapping column names to their indices.
        objective (str): Objective for the model.
        metrics (list): Metrics for the model.
        """

        self.model_name = "SAINT"
        self.column_idx = column_idx
        self.data_obj = data_obj

        if objective == 'binary':
            self.objective = "binary"
        elif objective == 'multi':
            self.objective = "multiclass"
        else:
            raise ValueError(f"Objective {objective} not supported")

        self.pred_dim = pred_dim
        self.input_dim = len(self.data_obj.feature_names)

        # Determine the number of heads (n_heads) by finding the first number from 8 to 1 that input_dim is divisible by
        for i in range(8, 0, -1):
            if self.input_dim % i == 0:
                self.n_heads = i
                break
        else:
            raise ValueError("No valid n_heads found. Ensure input_dim is divisible by at least one number between 1 and 8.")

        # Initialize the SAINT model
        self.model = SAINT(column_idx=self.column_idx,
                            cat_embed_input=None, 
                            cat_embed_dropout=None, 
                            use_cat_bias=None, 
                            cat_embed_activation=None, 
                            shared_embed=None, 
                            add_shared_embed=None, 
                            frac_shared_embed=None, 
                            continuous_cols=self.data_obj.feature_names, 
                            cont_norm_layer=None, 
                            embed_continuous_method='standard', 
                            cont_embed_dropout=None, 
                            cont_embed_activation=None, 
                            quantization_setup=None, 
                            n_frequencies=None, 
                            sigma=None, 
                            share_last_layer=None, 
                            full_embed_dropout=None, 
                            input_dim=self.input_dim, 
                            use_qkv_bias=False, 
                            n_heads=self.n_heads, 
                            n_blocks=2, 
                            attn_dropout=0.1, 
                            ff_dropout=0.2, 
                            ff_factor=4, 
                            transformer_activation='gelu', 
                            mlp_hidden_dims=[1024, 512, 256], 
                            mlp_activation=None, 
                            mlp_dropout=None, 
                            mlp_batchnorm=None, 
                            mlp_batchnorm_last=None, 
                            mlp_linear_first=None)
        self.wide_deep = WideDeep(deeptabular=self.model, pred_dim=self.pred_dim)
        
        self.trainer = Trainer(model=self.wide_deep, objective=self.objective, metrics=metrics)
        print(self.wide_deep)
        print(self.model)


    def to(self, device: torch.device):
        """
        Moves the model to the specified device.

        Parameters:
        device (torch.device): The device to move the model to.
        """
        self.wide_deep.to(device)

    

    def get_model(self):
        """
        Returns the initialized TabNet model for use in training or evaluation.
        
        Returns:
        model (TabNetClassifier): The TabNet classifier instance.
        """
        return self.wide_deep

    def fit(self, X_tab, target, n_epochs=65, batch_size=64):
        """
        Trains the TabNet model using the provided training data.
        
        Parameters:
        X_tab (Optional[ndarray], default: None) Input for the deeptabular model component. See pytorch_widedeep.preprocessing.TabPreprocessor
        target (Optional[ndarray], default: None) Target for the model.
        n_epochs (int, default: 65) Number of epochs to train the model.
        batch_size (int, default: 64) Number of samples per gradient update.      
        
        Returns:
        None
        """

        # print("we are here in fit")
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model.to(self.device)
        # # Ensure the model is in training mode
        # self.model.train()

        # # Convert target to zero-indexed
        # target = target - 1

        # # Create a DataLoader for the input data
        # dataset = TensorDataset(torch.tensor(X_tab), torch.tensor(target))
        # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # # Define the optimizer and loss function
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # criterion = torch.nn.CrossEntropyLoss()

        # print(f"Training on {self.device}")

        # # Training loop
        # for epoch in range(n_epochs):
        #     running_loss = 0.0
        #     for X_batch, y_batch in data_loader:
        #         # Move data to the appropriate device
        #         X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

        #         # Zero the parameter gradients
        #         optimizer.zero_grad()

        #         # Forward pass
        #         outputs = self.model(X_batch)
        #         loss = criterion(outputs, y_batch)

        #         # Backward pass and optimization
        #         loss.backward()
        #         optimizer.step()

        #         # Accumulate the loss
        #         running_loss += loss.item()

        #     # Print the average loss for this epoch
        #     print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {running_loss / len(data_loader):.4f}")

        print("we are here in fit")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Convert target to zero-indexed
        target = target - 1
        self.trainer.fit(X_tab=X_tab, target=target, n_epochs=n_epochs, batch_size=batch_size)

    
    def predict(self, X_test):
        """
        Makes predictions on the provided test data using the trained TabNet model.
        
        Parameters:
        X_test (array-like): Input features for testing.
        
        Returns:
        preds (array-like): Predicted labels for the test data.
        """
        
        return self.trainer.predict(X_tab=X_test)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Return logits for the input features
        # Convert numpy array to PyTorch tensor
        proba = self.trainer.predict_proba(X_tab=X)
        return torch.tensor(proba)
    
    def save_model(self, filepath):
        """
        Saves the trained TabNet model to the specified file path.
        
        Parameters:
        filepath (str): File path to save the model.
        
        Returns:
        None
        """
        torch.save(self.wide_deep.state_dict(), filepath)
    
    def load_model(self, filepath):
        """
        Loads a pre-trained TabNet model from the specified file path.
        
        Parameters:
        filepath (str): File path to load the model from.
        
        Returns:
        None
        """
        self.wide_deep.load_state_dict(torch.load(filepath))

    
    def eval(self):
        self.wide_deep.eval()



# Example usage:
# Initialize a TabNet model
# tabnet_model = TabNetModel(input_dim=54, output_dim=7)
# Train the model
# tabnet_model.fit(X_train, y_train, X_valid=X_val, y_valid=y_val)
# Get the trained model
# model = tabnet_model.get_model()
