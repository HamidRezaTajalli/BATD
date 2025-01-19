# ss.py
"""
Spectral Signature Defense for TabNet, SAINT, and FT-Transformer
===============================================================

This file contains:

1) Minimal dataset and DataLoader wrappers.
2) Feature-extraction stubs for TabNet, SAINT, and FT-Transformer.
3) The spectral_signature_defense() function.
4) The plotCorrelationScores() function for visualizing score distributions.
5) An example usage section at the bottom illustrating how to tie it all together.

Note:
 - You must adapt the "extract_features_*" functions to your actual model implementations.
 - Where you see 'model.forward_embeddings()' or 'forward_saint_features()' or
   'forward_ftt_features()', you can replace these calls with the correct logic or hooks
   for your own codebase.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


###############################################################################
# 1. Minimal Dataset Wrapper
###############################################################################
class SimpleDataset(Dataset):
    """
    A minimal PyTorch Dataset wrapper for (X, y) NumPy arrays.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return a tuple (inputs, label)
        return self.X[idx], self.y[idx]


###############################################################################
# 2. Feature-Extraction Stubs
###############################################################################
def extract_features_tabnet(
    tabnet_model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    device: torch.device = None
) -> np.ndarray:
    """
    Extract penultimate-layer (or suitable hidden) features from TabNet.

    Steps:
      1) Move model to `device`, put in eval mode.
      2) Create a DataLoader from X,y.
      3) For each batch, run a method that returns the hidden embedding
         (for instance, model.forward_embeddings(...)).
      4) Collect and return all embeddings as a single (N, D) array.

    IMPORTANT:
     - You must implement or reference the actual function/forward hook
       that obtains hidden embeddings from your TabNet model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tabnet_model.to(device)
    tabnet_model.eval()

    dataset = SimpleDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features = []

    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.float().to(device)
            # Hypothetical call that returns penultimate-layer features:
            # Replace with your actual approach (forward hook or direct method).
            features = tabnet_model.forward_embeddings(batch_x)  # <-- implement in your code
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def forward_saint_features(saint_model, X_batch: torch.Tensor) -> torch.Tensor:
    """
    Example function returning penultimate embeddings for SAINT.
    You need to replicate your SAINT forward pass but skip the final classifier MLP.
    
    (Pseudo-code; adapt to your code structure.)
    """
    # For demonstration, you might:
    #   1) separate cat/cont columns
    #   2) embed_data_mask(...)
    #   3) transformer -> reps
    #   4) return reps[:,0,:]
    # 
    # from .saint_lib.augmentations import embed_data_mask  # if needed
    # cat_idxs = saint_model.cat_idxs
    # con_idxs = saint_model.con_idxs
    # ...
    #
    # penultimate = reps[:, 0, :]
    # return penultimate
    raise NotImplementedError("Adapt this stub for your SAINT model.")


def extract_features_saint(
    saint_model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    device: torch.device = None
) -> np.ndarray:
    """
    Extract penultimate embeddings from a trained SAINT model in batch.

    You must define how exactly you retrieve the hidden representation
    (e.g., using forward_saint_features(...) or a forward hook).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saint_model.to(device)
    saint_model.eval()

    dataset = SimpleDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.float().to(device)
            # Acquire penultimate-layer features:
            penult_feats = forward_saint_features(saint_model, batch_x)
            all_features.append(penult_feats.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def forward_ftt_features(ftt_model, X_c: torch.Tensor, X_n: torch.Tensor) -> torch.Tensor:
    """
    Example function to return the penultimate representation from FT-Transformer,
    bypassing the final classification head.

    You must replicate the logic in your FT-Transformer forward but skip
    the final output layer (like ftt_model.head(...) if it exists).
    
    This is pseudo-code; adapt it to your library or forward pass.
    """
    raise NotImplementedError("Adapt this stub for your FT-Transformer.")


def extract_features_fttransformer(
    ft_model,
    X_c: np.ndarray,
    X_n: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    device: torch.device = None
) -> np.ndarray:
    """
    Batched penultimate feature extraction for FT-Transformer.
    X_c: categorical features
    X_n: continuous features
    y: labels
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ft_model.to(device)
    ft_model.eval()

    class FTDataset(Dataset):
        def __init__(self, Xc, Xn, y):
            self.Xc = Xc
            self.Xn = Xn
            self.y = y
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            return self.Xc[idx], self.Xn[idx], self.y[idx]

    dataset = FTDataset(X_c, X_n, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features = []
    with torch.no_grad():
        for batch_c, batch_n, _ in loader:
            batch_c = batch_c.to(device)
            batch_n = batch_n.float().to(device)
            feats = forward_ftt_features(ft_model, batch_c, batch_n)
            all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0)


###############################################################################
# 3. Spectral Signature Defense
###############################################################################
def spectral_signature_defense(
    features: np.ndarray,
    labels: np.ndarray,
    class_list,
    expected_poison_fraction: float = 0.05,
    extra_multiplier: float = 1.5,
):
    """
    Implements the Spectral Signature defense on extracted features.

    For each class c:
      1) Gather features of that class.
      2) Center them.
      3) Perform SVD -> top singular vector.
      4) Compute squared projection along that vector as score.
      5) Remove top K samples (K ~ expected_poison_fraction * extra_multiplier).
    
    Returns:
      - suspicious_indices (List[int]): indices flagged as suspicious
      - scores_by_class (Dict[int, np.ndarray]): arrays of scores for each class
    """
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must match in length.")

    suspicious_indices = []
    scores_by_class = {}
    all_indices = np.arange(len(labels))

    for c in class_list:
        # Indices for class c
        class_idxs = all_indices[labels == c]
        if len(class_idxs) < 2:
            # Not enough samples to do meaningful SVD
            continue

        class_feats = features[class_idxs]
        mean_feat = class_feats.mean(axis=0)
        centered = class_feats - mean_feat

        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        top_vec = Vt[0]

        # Score = squared projection
        scores = (centered @ top_vec) ** 2
        scores_by_class[c] = scores

        # Number of suspicious to remove
        K = int(len(class_feats) * expected_poison_fraction * extra_multiplier)
        K = min(K, len(class_feats))
        if K < 1:
            continue

        # The top-K scoring samples are suspicious
        suspicious_local = np.argsort(scores)[-K:]  # largest scores
        suspicious_global = class_idxs[suspicious_local]
        suspicious_indices.extend(suspicious_global)

    return suspicious_indices, scores_by_class


###############################################################################
# 4. Plot Correlation Scores
###############################################################################
def plotCorrelationScores(
    class_id: int,
    scores_for_class: np.ndarray,
    mask_for_class: np.ndarray,
    nbins: int = 100,
    label_clean: str = "Clean",
    label_poison: str = "Poisoned"
):
    """
    Plots a histogram of the correlation (spectral) scores for clean vs. poisoned
    or suspicious samples in a single class.

    Args:
      - class_id (int): The class label for the plot title
      - scores_for_class (np.ndarray): Array of shape (N_class_samples,) with the spectral scores
      - mask_for_class (np.ndarray): Boolean array of same shape, True means "poisoned" or "suspicious"
      - nbins (int): Number of bins for the histogram
      - label_clean (str): Legend label for clean distribution
      - label_poison (str): Legend label for poison distribution
    """
    plt.figure(figsize=(5,3))
    sns.set_style("white")
    sns.set_palette("tab10")

    scores_clean = scores_for_class[~mask_for_class]
    scores_poison = scores_for_class[mask_for_class]

    if len(scores_poison) == 0:
        # No poison => just plot clean
        if len(scores_clean) == 0:
            return  # no data
        bins = np.linspace(0, scores_clean.max(), nbins)
        plt.hist(scores_clean, bins=bins, color="green", alpha=0.75, label=label_clean)
    else:
        # We have both categories
        combined_max = max(scores_clean.max(), scores_poison.max())
        bins = np.linspace(0, combined_max, nbins)
        plt.hist(scores_poison, bins=bins, color="red", alpha=0.75, label=label_poison)
        plt.hist(scores_clean, bins=bins, color="green", alpha=0.75, label=label_clean)
        plt.legend(loc="upper right")

    plt.xlabel("Spectral Signature Score")
    plt.ylabel("Count")
    plt.title(f"Class {class_id}: Score Distribution")
    plt.show()


###############################################################################
# 5. Example Usage (Pseudocode)
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

    # Let's create a small synthetic example for demonstration:
    N = 1000
    D = 20
    X_demo = np.random.randn(N, D).astype(np.float32)
    y_demo = np.random.randint(0, 3, size=N)  # 3 classes: 0,1,2

    # Suppose we have "features" from a model. For demonstration, let's pretend
    # they are just X_demo. In reality, you call extract_features_*.
    features_demo = X_demo.copy()  # placeholder

    # Let's run spectral signature
    class_list = [0,1,2]
    suspicious_idx, scores_dict = spectral_signature_defense(
        features_demo,
        labels=y_demo,
        class_list=class_list,
        expected_poison_fraction=0.05,
        extra_multiplier=1.5
    )

    # Let's assume we have a "poisoned_mask" if we know which are truly poison.
    # For demonstration, create a random mask of 30 "poison" samples.
    poison_mask = np.zeros(N, dtype=bool)
    poison_indices = np.random.choice(N, size=30, replace=False)
    poison_mask[poison_indices] = True

    # Now we can plot the distribution for each class
    for c in class_list:
        # The scores for class c are in scores_dict[c].
        # The associated indices are where y_demo == c.
        class_idxs = np.where(y_demo == c)[0]
        class_scores = scores_dict[c]
        class_poison_mask = poison_mask[class_idxs]

        plotCorrelationScores(
            class_id=c,
            scores_for_class=class_scores,
            mask_for_class=class_poison_mask,
            nbins=50,
            label_clean="Clean",
            label_poison="Poisoned"
        )

    print("Done with Spectral Signature demo!")
