from pathlib import Path
import sys
import os
import csv

# Add the project root to the Python path
# Get the absolute path of the current file
current_file = Path(__file__).resolve()
# Get the project root (two directories up from the current file)
project_root = current_file.parent.parent.parent
# Add the project root to sys.path
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from src.attack import Attack




# importing all the models for this project
from src.models.FTT import FTTModel
from src.models.Tabnet import TabNetModel
from src.models.SAINT import SAINTModel
from src.models.CatBoost import CatBoostModel
from src.models.XGBoost import XGBoostModel

# importing all the datasets for this project
from src.dataset.BM import BankMarketing
from src.dataset.ACI import ACI
from src.dataset.HIGGS import HIGGS
from src.dataset.Diabetes import Diabetes
from src.dataset.Eye_Movement import EyeMovement
from src.dataset.KDD99 import KDD99
from src.dataset.CreditCard import CreditCard
from src.dataset.CovType import CovType  # Importing the ConvertCovType class from CovType.py


###############################################################################
# 2. Define a Function to Capture Transformer Activations for SAINT
###############################################################################
def get_transformer_activations_saint(saint_model,
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

            cls_activations = transformer_out[:, 0, :]  # shape [batch_size, dim]

            # ---------------------------------------------------------------
            # (2) # Collect the CLS embeddings for this batch
            # ---------------------------------------------------------------
            activations_list.append(cls_activations.cpu())

    # Concatenate over all batches -> shape [total_samples, n_features * dim]
    activations_all = torch.cat(activations_list, dim=0)
    return activations_all



###############################################################################
# 3. Compute Average Activation per Dimension & Identify Indices to Prune
###############################################################################
def compute_average_activations_saint(activations):
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


def get_prune_indices_saint(mean_activations, prune_ratio):
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
def create_pruning_mask_saint(total_dims, prune_inds, device):
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
    # mask = nn.Parameter(mask, requires_grad=False)
    return mask


class Pruned_mlpfory_saint(nn.Module):
    def __init__(self, pruning_mask, mlpfory):
        super().__init__()
        # Register the mask as a buffer (not trainable)
        self.register_buffer("pruning_mask", pruning_mask)
        self.mlpfory = mlpfory

    def forward(self, x):
        print(x.shape)
        print(self.pruning_mask.shape)
        print(x * self.pruning_mask)
        return self.mlpfory(x * self.pruning_mask)





def get_cls_activations_ftt(
    ftt_model,          # An instance of FTTModel
    dataset,            # A PyTorch Dataset yielding (X_c, X_n, y)
    batch_size=128,
    device=None,
    model_type="original"  
):
    """
    Given a trained FTTModel and a clean dataset, extracts the penultimate
    (CLS) embeddings for all samples. Returns a Tensor of shape [num_samples, dim].
    
    We will collect the output from FTTModel.forward_clstokens(...)
    for each batch of clean data.

    Args:
        ftt_model (FTTModel): your FTT model wrapper
        dataset (Dataset): yields (X_c, X_n, y)
        batch_size (int): batch size for collection
        device (torch.device): CPU/GPU device
        model_type (str): "original" or "converted"; whichever you are pruning

    Returns:
        activations_all (Tensor): [num_samples, dim], the CLS embeddings.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    ftt_model.to(device, model_type=model_type)
    ftt_model.eval(model_type=model_type)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    activations_list = []

    if model_type == 'original':

        # Collect [CLS] embeddings with no gradients
        with torch.no_grad():
            for X_c, X_n, _ in loader:
                X_c = X_c.to(device)
                X_n = X_n.to(device)

                # forward_clstokens outputs [batch_size, dim]
                cls_embeddings = ftt_model.forward_clstokens(X_c, X_n, model_type)
                activations_list.append(cls_embeddings.cpu())
    elif model_type == 'converted':
        with torch.no_grad():
            for X_n, _ in loader:
                X_c =torch.empty(X_n.shape[0], 0, dtype=torch.long)
                X_c, X_n = X_c.to(device), X_n.to(device)

                # forward clstokens outputs [batch_size, dim]
                cls_embeddings = ftt_model.forward_clstokens(X_c, X_n, model_type)
                activations_list.append(cls_embeddings.cpu())

    # Concatenate -> shape [total_samples, dim]
    activations_all = torch.cat(activations_list, dim=0)
    return activations_all




def compute_average_activations_ftt(activations):
    """
    Given [num_samples, dim] of absolute CLS activations,
    return a [dim] vector of average absolute activation.
    """
    return activations.abs().mean(dim=0)



def get_prune_indices_ftt(mean_activations, prune_ratio):
    """
    From mean_activations (shape [dim]), select
    the indices of the bottom prune_ratio fraction of dimensions.
    """
    dim = mean_activations.shape[0]
    num_prune = int(dim * prune_ratio)
    # sort ascending
    sorted_indices = torch.argsort(mean_activations)
    prune_inds = sorted_indices[:num_prune]
    return prune_inds



def create_pruning_mask_ftt(total_dim, prune_inds, device):
    """
    Returns a 1D mask with shape [dim],
    zeroing out the pruned dims.
    """
    mask = torch.ones(total_dim, dtype=torch.float32, device=device)
    mask[prune_inds] = 0.0
    return mask




class PrunedToLogitsFTT(nn.Module):
    """
    Replaces the FTTransformer's final 'to_logits' module, inserting a
    pruning mask after LN+ReLU but before the final Linear.
    """
    def __init__(self, original_to_logits, pruning_mask: torch.Tensor):
        super().__init__()
        # Expecting original_to_logits to be [LayerNorm, ReLU, Linear]
        if len(original_to_logits) != 3:
            raise ValueError("Unexpected structure of to_logits. Should be [LayerNorm, ReLU, Linear].")

        self.ln = original_to_logits[0]
        self.act = original_to_logits[1]
        self.linear = original_to_logits[2]

        # Register the mask as a buffer (not trainable)
        self.register_buffer("pruning_mask", pruning_mask)

    def forward(self, x):
        # Now apply the pruning mask
        x = x * self.pruning_mask

        # Normal FTTransformer final sequence
        x = self.ln(x)
        x = self.act(x)
        # Then final classification layer
        x = self.linear(x)
        return x




def get_tabnet_penultimate_features(tabnet_model, dataset, device=None, batch_size=256):
    """
    Collect the penultimate representations (res) from a TabNet model on a CLEAN dataset.
    
    Args:
        tabnet_model: A trained TabNetClassifier instance from pytorch_tabnet.
        dataset:      A PyTorch Dataset yielding (X, y) or just X for inference.
        device:       CPU or GPU device to run on.
        batch_size:   Batch size for the data loader.
    
    Returns:
        A Tensor of shape [num_samples, n_d] containing the final 'res' 
        (the sum of step outputs) for each sample, BEFORE final_mapping.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tabnet_model.network.eval()  # put TabNet in eval mode
    tabnet_model.network.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # We'll store the penultimate features for all samples here
    all_penultimate = []

    with torch.no_grad():
        for batch_data in loader:
            # If dataset yields (X, y), separate them; otherwise, batch_data is X
            if isinstance(batch_data, (list, tuple)):
                X, *_ = batch_data
            else:
                X = batch_data

            X = X.to(device).float()

            # 1) We intercept the forward at the point of "res"
            #    We'll replicate the logic inside TabNetNoEmbeddings:
            #    res = sum(steps_output), out = final_mapping(res)
            #    We want `res` but skip final_mapping for now.

            # EXACT procedure requires we break open TabNet. We'll do:
            embedded_x = tabnet_model.network.embedder(X)       # => shape [batch, embedding_dim]
            # Then call the tabnet block without final_mapping
            res, M_loss = forward_tabnet_no_final(tabnet_model.network.tabnet, embedded_x)

            # Now 'res' is shape [batch_size, n_d], the penultimate representation
            all_penultimate.append(res.cpu())

    penultimate_tensor = torch.cat(all_penultimate, dim=0)
    return penultimate_tensor


def forward_tabnet_no_final(tabnet_block, embedded_x):
    """
    A helper that runs TabNetNoEmbeddings encoder steps to produce 
    the aggregated representation 'res' before final_mapping.

    tabnet_block is an instance of TabNetNoEmbeddings:
       out, M_loss = tabnet_block(embedded_x)

    But we skip `final_mapping`. 
    Instead, we manually do the steps up to obtaining `res`.

    Returns:
        res: shape [batch, n_d]
        M_loss: whatever TabNet uses for the sparse loss (optional)
    """
    # The forward in TabNetNoEmbeddings is something like:
    #   steps_output, M_loss = self.encoder(x)
    #   res = torch.sum(torch.stack(steps_output, dim=0), dim=0)
    #   out = self.final_mapping(res)
    #   return out, M_loss
    #
    # We only want 'res' here.

    steps_output, M_loss = tabnet_block.encoder(embedded_x)
    # sum up step outputs
    res = torch.sum(torch.stack(steps_output, dim=0), dim=0)
    # skip final_mapping
    return res, M_loss



def compute_average_activations_tabnet(activations):
    """
    Given activations of shape [num_samples, n_d],
    return a vector [n_d] of mean absolute activations.
    """
    return activations.abs().mean(dim=0)

def get_prune_indices_tabnet(mean_activations, prune_ratio):
    """
    From mean_activations [n_d], pick the bottom fraction to prune.
    """
    dim = mean_activations.shape[0]
    num_prune = int(dim * prune_ratio)
    sorted_inds = torch.argsort(mean_activations)
    prune_inds = sorted_inds[:num_prune]
    return prune_inds

def create_pruning_mask_tabnet(n_d, prune_inds, device):
    """
    Return a 1D mask of shape [n_d],
    where pruned dims are 0.0
    """
    mask = torch.ones(n_d, dtype=torch.float32, device=device)
    mask[prune_inds] = 0.0
    return mask



class PrunedFinalMappingTabNet(nn.Module):
    """
    Wraps TabNetNoEmbeddings so that we can multiply 'res' by a pruning mask
    before the final linear mapping.
    """
    def __init__(self, tabnet_block, pruning_mask):
        super().__init__()
        # tabnet_block is an instance of TabNetNoEmbeddings
        self.final_mapping = tabnet_block.final_mapping
        self.pruning_mask = nn.Parameter(pruning_mask, requires_grad=False)

    def forward(self, res):
        
        # apply mask
        res = res * self.pruning_mask
        out = self.final_mapping(res)
        return out







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
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lambd", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--exp_num", type=int, default=0)
    parser.add_argument("--prune_rate", type=float, default=0.9)

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
    logging.info(f"Pruning with the following arguments: {args}")
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


    # create the experiment results directory
    results_path = Path("./results/pruning")
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    csv_file_address = results_path / Path(f"{dataset_name}.csv")
    if not csv_file_address.exists():
        csv_file_address.touch()
        
        csv_header = ['EXP_NUM', 'DATASET', 'MODEL', 'TARGET_LABEL', 'EPSILON', 'PRUNE_RATE', 'MU', 'BETA', 'LAMBDA', 'P_CDA', 'P_ASR', 'FP_CDA', 'FP_ASR']
        # insert the header row into the csv file
        with open(csv_file_address, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)


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

    if model_name == "ftt" and data_obj.cat_cols:
        model.load_model(poisoned_model_address_toload, model_type="original")  
        model.to(device, model_type="original")
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



    if model_name == "saint":
        print("==> [Step 1] Gathering Transformer Activations on Clean Data...")
        activations_all = get_transformer_activations_saint(model, clean_trainset, device=device)
        print("    Activations shape:", activations_all.shape)  # e.g., [N, nFeats*dim]

        print("==> [Step 2] Computing Mean Activations + Identifying Pruning Indices...")
        mean_acts = compute_average_activations_saint(activations_all)  # shape [dim]
        prune_inds = get_prune_indices_saint(mean_acts, prune_rate)     # dims to prune
        print(f"    Total dims: {mean_acts.numel()}, Pruning {len(prune_inds)} dims ({prune_rate*100}%).")


        print("==> [Step 3] inject pruning mask into the model ...")

        mask = create_pruning_mask_saint(mean_acts.numel(), prune_inds, device)
        pruned_mlpfory = Pruned_mlpfory_saint(mask, model.model.mlpfory)
        model.model.mlpfory = pruned_mlpfory
        model.opt.epochs = 5

        # for name, param in model.model.named_parameters():
        #     if "mlpfory.mlpfory.layers.2" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
    
    elif model_name == "ftt":

        model_type = "original" if data_obj.cat_cols else "converted"

        # Step 1: Gather [CLS] embeddings on clean data
        print("==> [Step 1] Gathering [CLS] Activations on Clean Data...")
        cls_activations = get_cls_activations_ftt(
            model, clean_trainset, batch_size=128, device=device, model_type=model_type
        )
        print("    CLS Activations shape:", cls_activations.shape)  # e.g. [N, dim]

        # Step 2: Compute mean activation and prune indices
        print("==> [Step 2] Computing Mean Activations + Identifying Pruning Indices...")
        mean_acts = compute_average_activations_ftt(cls_activations)  # [dim]
        prune_inds = get_prune_indices_ftt(mean_acts, prune_rate)
        print(f"    total dims: {mean_acts.numel()}, prune {len(prune_inds)} dims ({prune_rate*100}%).")

        # Step 3: Create a pruning mask
        mask = create_pruning_mask_ftt(mean_acts.numel(), prune_inds, device)
        print("==> [Step 3] Created pruning mask of shape:", mask.shape)

        # Step 4: Replace the final tologit module with a pruned version
        print("==> Replacing final tologit module with pruned version...")
        if data_obj.cat_cols:
            model.model_original.to_logits = PrunedToLogitsFTT(model.model_original.to_logits, mask)
        else: 
            model.model_converted.to_logits = PrunedToLogitsFTT(model.model_converted.to_logits, mask)
        model.epochs = 5

        # model_to_freeze = model.model_original if data_obj.cat_cols else model.model_converted
        # for name, param in model_to_freeze.named_parameters():
        #     if "to_logits.linear" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

    elif model_name == "tabnet":

        # Step 1: get penultimate features
        penultimate_acts = get_tabnet_penultimate_features(model.model, clean_trainset, device=device)
        print("Penultimate shape:", penultimate_acts.shape)


        # Step 2: average activation + prune
        mean_acts = compute_average_activations_tabnet(penultimate_acts)
        prune_inds = get_prune_indices_tabnet(mean_acts, prune_rate)
        mask = create_pruning_mask_tabnet(mean_acts.shape[0], prune_inds, device)
        print(f"Pruning {len(prune_inds)}/{mean_acts.shape[0]} dims ({100*prune_rate}%).")

        # Step 3: inject mask
        old_tabnet_block = model.model.network.tabnet
        pruned_final_mapping = PrunedFinalMappingTabNet(old_tabnet_block, mask)
        model.model.network.tabnet.final_mapping = pruned_final_mapping
        model.max_epochs = 5

        # for name, param in model.model.network.named_parameters():
        #     if "tabnet.final_mapping.final_mapping.weight" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

    else:
        raise ValueError(f"Model {model_name} is not supported for pruning.")


    # test the pruned model on clean data and poisoned data
    converted = False if FTT and data_obj.cat_cols else True    
    p_asr = attack.test(reverted_poisoned_testset, converted=converted)
    p_cda = attack.test(clean_testset, converted=converted)


    print("==> [Step 4] Fine-Tuning the pruned model on clean data...")
    #Train the model on the poisoned training dataset
    
    attack.train(clean_dataset, converted=converted)
    logging.info("=== Fine-Tuning Completed ===")

    print("==> [Step 5] Testing the pruned model on clean data and poisoned data...")
    fp_asr = attack.test(reverted_poisoned_testset, converted=converted)
    fp_cda = attack.test(clean_testset, converted=converted)
    print("=== Testing Completed ===")


    # save the results to the csv file
    with open(csv_file_address, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([args.exp_num, dataset_name, model_name, target_label, epsilon, prune_rate, mu, beta, lambd, p_cda, p_asr, fp_cda, fp_asr])


    # creating and checking the path for saving and loading the pruned models.
    pruned_model_path = models_path / Path(f"pruned")
    if not pruned_model_path.exists():
        pruned_model_path.mkdir(parents=True, exist_ok=True)
    pruned_model_address = pruned_model_path / Path(f"{model_name}_{data_obj.dataset_name}_{target_label}_{mu}_{beta}_{lambd}_{epsilon}_pruned_model.pth")

    # Save the poisoned model with Unix timestamp in the filename
    if model_name == "ftt" and data_obj.cat_cols:
        attack.model.save_model(pruned_model_address, model_type="original")
    else:
        attack.model.save_model(pruned_model_address)







# TODO: small portion of the dataset should be used for fine-tuning phase.
# TODO: check the effect of mask on saint