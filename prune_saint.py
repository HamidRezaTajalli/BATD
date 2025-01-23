from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from attack import Attack




# importing all the models for this project
from models.FTT import FTTModel
from models.Tabnet import TabNetModel
from models.SAINT import SAINTModel
from models.CatBoost import CatBoostModel
from models.XGBoost import XGBoostModel

# importing all the datasets for this project
from dataset.BM import BankMarketing
from dataset.ACI import ACI
from dataset.HIGGS import HIGGS
from dataset.Diabetes import Diabetes
from dataset.Eye_Movement import EyeMovement
from dataset.KDD99 import KDD99
from dataset.CreditCard import CreditCard
from dataset.CovType import CovType  # Importing the ConvertCovType class from CovType.py


###############################################################################
# 2. Define a Function to Capture Transformer Activations
###############################################################################
def get_transformer_activations(saint_model,
    dataset,
    batch_size: int = 128,
    device: torch.device = None):
    """
    Given a SAINT model and a DataLoader of clean samples,
    returns a Tensor of shape [total_samples, (num_features * dim)]
    containing the final Transformer-layer activations before the final MLP.

    Args:
        model  (SAINT):    A SAINT (or similar) Transformer-based model.
        loader (DataLoader): A PyTorch DataLoader that yields (x_categ, x_cont, y).
        device (torch.device): CPU or GPU device.

    Returns:
        activations_all (Tensor): The collected activations from the final
                                  Transformer output, flattened per sample.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saint_model.to(device)
    saint_model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    activations_list = []

    # Disable gradient calculations for activation collection
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            transformer_out = saint_model.forward_reps(batch_x)
            # "transformer_out" has shape [batch_size, (n_categ + n_cont), dim]

            # Flatten out the last two dims to create [batch_size, (n_features * dim)]
            # n_features = n_categ + n_cont
            batch_activations = transformer_out.reshape(transformer_out.size(0), -1)

            # ---------------------------------------------------------------
            # (2) Collect activations into a list for all samples
            # ---------------------------------------------------------------
            activations_list.append(batch_activations.cpu())

    # Concatenate over all batches -> shape [total_samples, n_features * dim]
    activations_all = torch.cat(activations_list, dim=0)
    return activations_all



###############################################################################
# 3. Compute Average Activation per Dimension & Identify Indices to Prune
###############################################################################
def compute_average_activations(activations):
    """
    Given a 2D activation matrix of shape [num_samples, num_dimensions],
    returns a 1D tensor of shape [num_dimensions] containing the average
    absolute activation across all samples.

    Args:
        activations (Tensor): shape [total_samples, total_dims]

    Returns:
        mean_acts (Tensor): shape [total_dims], mean absolute activation
    """
    # Compute mean absolute activation for each dimension
    # across all samples (dim=0).
    mean_acts = activations.abs().mean(dim=0)  # [total_dims]
    return mean_acts


def get_prune_indices(mean_activations, prune_ratio):
    """
    From a vector of mean activations, get the indices of the lowest
    prune_ratio fraction of dimensions.

    Args:
        mean_activations (Tensor): shape [total_dims]
        prune_ratio (float): fraction of dimensions to prune, e.g. 0.2

    Returns:
        prune_inds (LongTensor): indices of the dimensions to prune
    """
    total_dims = mean_activations.shape[0]
    num_prune = int(prune_ratio * total_dims)

    # Sort dimensions by ascending activation
    sorted_inds = torch.argsort(mean_activations)  # ascending
    # Select the lowest-activation indices
    prune_inds = sorted_inds[:num_prune]
    return prune_inds


###############################################################################
# 4. Create a "Pruning Mask" and Integrate It into the Model
###############################################################################
def create_pruning_mask(total_dims, prune_inds, device):
    """
    Create a 1D pruning mask of length 'total_dims' that is 1.0
    for unpruned dimensions and 0.0 for pruned dimensions.

    Args:
        total_dims (int): total number of dimensions in the final Transformer output
        prune_inds (Tensor): indices of dimensions to prune
        device (torch.device)

    Returns:
        mask (nn.Parameter): shape [total_dims], with 1/0 entries
    """
    mask = torch.ones(total_dims, dtype=torch.float32, device=device)
    mask[prune_inds] = 0.0
    # We wrap this in nn.Parameter so it's easy to use in forward, but
    # we set requires_grad=False because we don't want to train it.
    mask = nn.Parameter(mask, requires_grad=False)
    return mask


class PrunedSAINT(nn.Module):
    """
    A wrapper around the SAINT model that applies a dimension-wise
    masking to the final Transformer output before the final MLP.

    This approach does not physically remove dimensions from the weight
    matrices, but sets them to zero via a mask. This effectively "prunes"
    those neurons.

    You could also apply a more advanced approach that physically
    modifies weight shapes, but masking is simpler to implement.
    """
    def __init__(self, saint_model):
        super().__init__()
        self.model = saint_model
        self.register_parameter("pruning_mask", None)  # init as None, set later

    def set_pruning_mask(self, mask):
        """
        Register a pruning mask for the penultimate representation.
        mask: nn.Parameter of shape [n_features * dim].
        """
        # Officially register the mask as a parameter (though it's not trainable).
        self.pruning_mask = mask

    # def forward(self, x_categ, x_cont):
    #     """
    #     1) Forward pass to get Transformer output
    #     2) Apply pruning mask
    #     3) Pass the masked representation to the final MLP
    #     4) Return the model output (logits, etc.)
    #     """
    #     # (A) Get the final Transformer output from the original SAINT:
    #     #     We'll mimic a partial forward. SAINT typically does something like:
    #     #     transformer_out = self.model.transformer(...)
    #     #     flatten_x = transformer_out.reshape(batch_size, -1)
    #     #     out = self.model.mlp(flatten_x)
    #     #
    #     # For clarity, let's explicitly call the model's forward.
    #     # But you need to ensure your SAINT forward uses the same shape logic.
    #     #
    #     # If your original SAINT forward requires x_categ_enc, x_cont_enc, etc.,
    #     # you can replicate that here or modify SAINT to separate the steps.
    #     transformer_out = self.model.transformer(
    #         self.model.embeds(x_categ),
    #         get_continuous_embeddings(self.model, x_cont)
    #     )
    #     # shape is [batch_size, (n_categ + n_cont), dim]

    #     # (B) Flatten the representation
    #     bsz = transformer_out.size(0)
    #     flatten_out = transformer_out.view(bsz, -1)  # shape [batch_size, total_dims]

    #     # (C) Apply pruning mask if it exists
    #     if self.pruning_mask is not None:
    #         flatten_out = flatten_out * self.pruning_mask  # elementwise multiplication

    #     # (D) Pass to the final MLP for classification/regression
    #     out = self.model.mlp(flatten_out)

    #     return out


# ###############################################################################
# # 5. Fine-Tuning After Pruning
# ###############################################################################
# def fine_tune(model, loader, device, epochs=5, lr=1e-4):
#     """
#     Optionally fine-tune the pruned model on the clean dataset
#     to restore accuracy lost due to pruning.

#     Args:
#         model (nn.Module): Pruned SAINT model (or wrapper).
#         loader (DataLoader): Clean data loader.
#         device (torch.device)
#         epochs (int): number of fine-tuning epochs
#         lr (float): learning rate
#     """
#     model.train()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()  # For classification, adjust as needed

#     for epoch in range(epochs):
#         total_loss = 0.0
#         for x_categ, x_cont, labels in loader:
#             x_categ, x_cont, labels = x_categ.to(device), x_cont.to(device), labels.to(device)

#             optimizer.zero_grad()
#             logits = model(x_categ, x_cont)  # forward pass
#             loss = criterion(logits, labels)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#         avg_loss = total_loss / len(loader)
#         print(f"[Fine-Tune] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


# ###############################################################################
# # 6. Putting It All Together (Example Workflow)
# ###############################################################################
# def fine_pruning_example(
#     saint_model,
#     clean_data_loader,
#     prune_ratio=0.2,
#     fine_tune_epochs=5,
#     device="cuda"
# ):
#     """
#     Demonstrates how to perform Fine-Pruning on a SAINT model using
#     a small set of clean data.

#     1) Collect final Transformer activations
#     2) Compute average activation per dimension
#     3) Prune the low-activation dimensions
#     4) (Optional) Fine-tune on clean data

#     Args:
#         saint_model (SAINT): your SAINT model instance (unpruned).
#         clean_data_loader (DataLoader): DataLoader with CLEAN data only.
#         prune_ratio (float): fraction of dimensions to prune.
#         fine_tune_epochs (int): how many epochs to fine-tune.
#         device (str or torch.device)
#     """
#     device = torch.device(device if torch.cuda.is_available() else "cpu")
#     saint_model.to(device)

#     print("==> [Step 1] Gathering Transformer Activations on Clean Data...")
#     activations_all = get_transformer_activations(saint_model, clean_data_loader, device)
#     print("    Activations shape:", activations_all.shape)  # e.g., [N, nFeats*dim]

#     print("==> [Step 2] Computing Mean Activations and Sorting...")
#     mean_acts = compute_average_activations(activations_all)
#     prune_inds = get_prune_indices(mean_acts, prune_ratio)
#     print(f"    Total dimensions: {mean_acts.numel()}, Pruning {len(prune_inds)} dims ({prune_ratio*100}%).")

#     print("==> [Step 3] Creating Pruned Model Wrapper + Mask...")
#     pruned_model = PrunedSAINT(saint_model).to(device)
#     mask = create_pruning_mask(mean_acts.numel(), prune_inds, device)
#     pruned_model.set_pruning_mask(mask)

#     # Evaluate or measure performance on clean data + (if possible) backdoor
#     # before fine-tuning, to see how much accuracy was lost.
#     # Here, we won't show a full evaluation function, but you'd typically do:
#     # eval_accuracy = evaluate(pruned_model, clean_data_loader, device)
#     # print("Accuracy after pruning (no fine-tune):", eval_accuracy)

#     print("==> [Step 4] Fine-tuning the pruned model on clean data...")
#     fine_tune(pruned_model, clean_data_loader, device, epochs=fine_tune_epochs, lr=1e-4)

#     # Evaluate again if needed
#     # final_accuracy = evaluate(pruned_model, clean_data_loader, device)
#     # print("Final accuracy after fine-tuning:", final_accuracy)

#     return pruned_model


# ###############################################################################
# # 7. Usage Example (Pseudo-Code)
# ###############################################################################
# if __name__ == "__main__":
#     # -------------------------------------------------------------------------
#     # 7.1 Create or Load a Pretrained SAINT Model (potentially backdoored)
#     # -------------------------------------------------------------------------
#     # For instance:
#     categories = [10, 5, 7]  # example
#     num_continuous = 4
#     saint = SAINT(
#         categories=categories,
#         num_continuous=num_continuous,
#         dim=32,
#         depth=2,
#         heads=4,
#         dim_head=16,
#         dim_out=10,  # e.g., 10-class classification
#         cont_embeddings='MLP',
#         attentiontype='col'
#     )

#     # Assume 'saint' is loaded from a checkpoint or already trained.
#     # e.g. saint.load_state_dict(torch.load("backdoored_saint.pth"))

#     # -------------------------------------------------------------------------
#     # 7.2 Build a DataLoader with CLEAN data only
#     # -------------------------------------------------------------------------
#     # This is just a skeleton. In reality, you'd load your real dataset.
#     import random
#     from torch.utils.data import DataLoader, TensorDataset

#     def random_clean_data(num_samples=256, cat_dim=3, cont_dim=4, num_classes=10):
#         """
#         Return random (x_categ, x_cont, labels) just for illustration.
#         """
#         x_categ = torch.randint(0, 10, (num_samples, cat_dim))  # random cat
#         x_cont  = torch.randn(num_samples, cont_dim)            # random cont
#         labels  = torch.randint(0, num_classes, (num_samples,)) # random labels
#         return x_categ, x_cont, labels

#     x_categ, x_cont, labels = random_clean_data(num_samples=512)
#     clean_dataset = TensorDataset(x_categ, x_cont, labels)
#     clean_data_loader = DataLoader(clean_dataset, batch_size=32, shuffle=True)

#     # -------------------------------------------------------------------------
#     # 7.3 Perform Fine-Pruning
#     # -------------------------------------------------------------------------
#     pruned_saint = fine_pruning_example(
#         saint_model       = saint,
#         clean_data_loader = clean_data_loader,
#         prune_ratio       = 0.2,       # prune 20% of dims
#         fine_tune_epochs  = 3,         # do 3 epochs of fine-tuning
#         device            = "cuda"     # or "cpu"
#     )

#     print("==> Fine-Pruning complete. You now have a pruned SAINT model.")








###############################################################################
# 5. Main function
###############################################################################
if __name__ == "__main__":
    """
    In practice, you would:
      1) Train or load your model (TabNet, SAINT, or FT-Transformer).
      2) Extract penultimate features via the appropriate function.
      3) Run spectral_signature_defense(...) to identify suspicious samples.
      4) If you have a ground-truth poison mask, or if you want to treat
         suspicious indices as "poison", you can plot with plotCorrelationScores.
    """


    # set up an argument parser
    import argparse
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--mu", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lambd", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--exp_num", type=int, default=0)
    parser.add_argument("--prune_rate", type=float, default=0.5)

    # parse the arguments
    args = parser.parse_args()

    available_datasets = ["aci", "bm", "higgs", "diabetes", "eye_movement", "kdd99", "credit_card", "covtype"]
    available_models = ["ftt", "tabnet", "saint"]

    # check and make sure that the dataset and model are available
    # the dataset and model names are case insensitive, so compare the lowercase versions of the arguments
    if args.dataset_name.lower() not in available_datasets:
        raise ValueError(f"Dataset {args.dataset_name} is not available. Please choose from: {available_datasets}")
    if args.model_name.lower() not in available_models:
        raise ValueError(f"Model {args.model_name} is not available. Please choose from: {available_models}")

     
     # check and make sure that mu, beta, lambd, and epsilon are within the valid range
    if args.mu < 0 or args.mu > 1:
        raise ValueError(f"Mu must be between 0 and 1. You provided: {args.mu}")
    if args.beta < 0 or args.beta > 1:
        raise ValueError(f"Beta must be between 0 and 1. You provided: {args.beta}")
    if args.lambd < 0 or args.lambd > 1:
        raise ValueError(f"Lambd must be between 0 and 1. You provided: {args.lambd}")
    if args.epsilon < 0 or args.epsilon > 1:
        raise ValueError(f"Epsilon must be between 0 and 1. You provided: {args.epsilon}")
    
    if args.prune_rate < 0 or args.prune_rate > 1:
        raise ValueError(f"Pruning rate must be between 0 and 1. You provided: {args.prune_rate}")
    

    # log all the arguments before running the experiment
    print("-"*100)
    logging.info(f"Defending with the following arguments: {args}")
    print("-"*100)
    print("\n")

    model_name = args.model_name
    dataset_name = args.dataset_name
    target_label = args.target_label
    mu = args.mu
    beta = args.beta
    lambd = args.lambd
    epsilon = args.epsilon
    prune_rate = args.prune_rate


    dataset_dict = {
        "aci": ACI,
        "bm": BankMarketing,
        "higgs": HIGGS,
        "diabetes": Diabetes,
        "eye_movement": EyeMovement,
        "kdd99": KDD99,
        "credit_card": CreditCard,
        "covtype": CovType
    }

    model_dict = {
        "ftt": FTTModel,
        "tabnet": TabNetModel,
        "saint": SAINTModel,
        "catboost": CatBoostModel,
        "xgboost": XGBoostModel
    }


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Step 1: Initialize the dataset object which can handle, convert and revert the dataset.
    data_obj = dataset_dict[dataset_name]()


    models_path = Path("./saved_models")

    # creating and checking the path for saving and loading the poisoned models.
    poisoned_model_path = models_path / Path(f"poisoned")
    if not poisoned_model_path.exists():
        poisoned_model_path.mkdir(parents=True, exist_ok=True)
    poisoned_model_address = poisoned_model_path / Path(f"{model_name}_{data_obj.dataset_name}_{target_label}_{mu}_{beta}_{lambd}_{epsilon}_poisoned_model.pth")


    # if the model name is xgboost, catboost, or tabnet, then we should add .zip at the end of the model name.
    if model_name in ['tabnet', 'xgboost', 'catboost']:
        # add .zip at the end of the model name in addition to the existing suffix
        poisoned_model_address_toload = poisoned_model_address.with_suffix(poisoned_model_address.suffix + ".zip")
    else:
        poisoned_model_address_toload = poisoned_model_address

    if not poisoned_model_address_toload.exists():
        raise ValueError(f"Poisoned model address at {poisoned_model_address_toload} does not exist.")
    


    
    if model_name == "ftt":
        model = FTTModel(data_obj=data_obj)
    elif model_name == "tabnet":
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
    elif model_name == "saint":
        model = SAINTModel(data_obj=data_obj, is_numerical=False)

    if model_name == "ftt":
        model.load_model(poisoned_model_address_toload, model_type="original")
    else:
        model.load_model(poisoned_model_address_toload)
    model.to(device)


    attack = Attack(device=device, model=model, data_obj=data_obj, target_label=target_label, mu=mu, beta=beta, lambd=lambd, epsilon=epsilon)

    attack.load_poisoned_dataset()

    poisoned_trainset, poisoned_testset = attack.poisoned_dataset
    poisoned_train_samples, poisoned_test_samples = attack.poisoned_samples
    attack.poisoned_dataset = (poisoned_trainset, poisoned_testset)
    attack.poisoned_samples = (poisoned_train_samples, poisoned_test_samples)


    # Assuming poisoned_trainset and poisoned_train_samples are PyTorch datasets or tensors
    poisoned_indices = []


    # Convert poisoned_train_samples to a set for faster lookup
    poisoned_samples_set = set(tuple(sample.tolist()) for sample, _ in poisoned_train_samples)

    # Iterate over poisoned_trainset to find indices of poisoned samples
    for idx, (sample, _) in enumerate(poisoned_trainset):
        # Convert the sample to a tuple for comparison
        sample_tuple = tuple(sample.tolist())
        
        # Check if the sample is in the poisoned_samples_set
        if sample_tuple in poisoned_samples_set:
            poisoned_indices.append(idx)

    # poisoned_indices now contains the indices of poisoned samples in poisoned_trainset
    # print("Indices of poisoned samples:", poisoned_indices)

    # Step 11: Revert the poisoned dataset to the original categorical features
    FTT = True if attack.model.model_name == "FTTransformer" else False
    if data_obj.cat_cols:
        reverted_poisoned_trainset = attack.data_obj.Revert(attack.poisoned_dataset[0], FTT=FTT)
        reverted_poisoned_testset = attack.data_obj.Revert(attack.poisoned_dataset[1], FTT=FTT)
    else:
        reverted_poisoned_trainset = attack.poisoned_dataset[0]
        reverted_poisoned_testset = attack.poisoned_dataset[1]

    reverted_poisoned_dataset = (reverted_poisoned_trainset, reverted_poisoned_testset)

    # get the clean train and test datasets
    if FTT and data_obj.cat_cols:
        clean_dataset = attack.data_obj.get_normal_datasets_FTT()
    else:
        clean_dataset = attack.data_obj.get_normal_datasets()
    clean_trainset = clean_dataset[0]
    clean_testset = clean_dataset[1]


    print("==> [Step 1] Gathering Transformer Activations on Clean Data...")
    activations_all = get_transformer_activations(model, clean_trainset, device)
    print("    Activations shape:", activations_all.shape)  # e.g., [N, nFeats*dim]

    exit()

