import torch
from torch import nn
from models import SAINT

# data_openml.py contains functions to download and preprocess the OpenML dataset
from data_openml import data_prep_openml, task_dset_ids, DataSetCatCon

import argparse
from torch.utils.data import DataLoader
import torch.optim as optim

# utils.py contains helper functions such as counting parameters, calculating classification scores, etc.
from utils import count_parameters, classification_scores, mean_sq_error

# augmentations.py contains data augmentation methods
from augmentations import embed_data_mask
from augmentations import add_noise

import os
import numpy as np

def argparse_prepare():


    # -------------------
    # ARGUMENT PARSER SETUP
    # -------------------
    
    parser = argparse.ArgumentParser()

    # # --dset_id is the OpenML dataset ID to load; required argument
    # parser.add_argument('--dset_id', required=True, type=int)

    # --vision_dset is a flag (boolean) that indicates if this is a vision dataset
    parser.add_argument('--vision_dset', action='store_true')

    # --task indicates the type of problem: 'binary', 'multiclass', or 'regression'; required
    parser.add_argument('--task', required=False, type=str, choices=['binary','multiclass','regression'])

    # --cont_embeddings chooses how continuous features are embedded. Options:
    # 'MLP', 'Noemb', 'pos_singleMLP'
    parser.add_argument('--cont_embeddings', default='MLP', type=str, choices=['MLP','Noemb','pos_singleMLP'])

    # --embedding_size is the dimensionality of embeddings for features
    parser.add_argument('--embedding_size', default=32, type=int)

    # --transformer_depth is the number of Transformer encoder layers
    parser.add_argument('--transformer_depth', default=6, type=int)

    # --attention_heads is the number of attention heads in the Multi-Head Attention
    parser.add_argument('--attention_heads', default=8, type=int)

    # --attention_dropout is the dropout applied to attention probabilities
    parser.add_argument('--attention_dropout', default=0.1, type=float)

    # --ff_dropout is the dropout applied in the feed-forward blocks
    parser.add_argument('--ff_dropout', default=0.1, type=float)

    # --attentiontype controls the type of attention used, e.g., 'colrow', 'col', etc.
    parser.add_argument('--attentiontype', default='colrow', type=str,
        choices=['col','colrow','row','justmlp','attn','attnmlp'])

    # --optimizer chooses the optimizer type, e.g., AdamW, Adam, or SGD
    parser.add_argument('--optimizer', default='AdamW', type=str, choices=['AdamW','Adam','SGD'])

    # --scheduler chooses how the learning rate is scheduled, either 'cosine' or 'linear'
    parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine','linear'])

    # --lr is the learning rate
    parser.add_argument('--lr', default=0.0001, type=float)

    # --epochs is the total number of epochs for training
    parser.add_argument('--epochs', default=100, type=int)

    # --batchsize is the batch size used in the DataLoader
    parser.add_argument('--batchsize', default=256, type=int)

    # --savemodelroot is the folder in which to save the trained model checkpoints
    parser.add_argument('--savemodelroot', default='./bestmodels', type=str)

    # --run_name is a string label for the run (helpful when logging with wandb)
    parser.add_argument('--run_name', default='testrun', type=str)

    # --set_seed sets a random seed for reproducibility
    parser.add_argument('--set_seed', default=1, type=int)

    # --dset_seed is another seed for dataset splitting
    parser.add_argument('--dset_seed', default=5, type=int)

    # --active_log is a flag indicating whether to log metrics to wandb
    parser.add_argument('--active_log', action='store_true')

    # --pretrain is a flag to indicate if the model should be pretrained
    parser.add_argument('--pretrain', action='store_true')

    # --pretrain_epochs is the number of epochs to pretrain
    parser.add_argument('--pretrain_epochs', default=50, type=int)

    # --pt_tasks lists the self-supervised learning tasks used during pretraining (contrastive, denoising, etc.)
    parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str, nargs='*',
        choices=['contrastive','contrastive_sim','denoising'])

    # --pt_aug indicates which augmentations (e.g., mixup, cutmix) to apply during pretraining
    parser.add_argument('--pt_aug', default=[], type=str, nargs='*', choices=['mixup','cutmix'])

    # --pt_aug_lam is a hyperparameter for the augmentations' lamda value in pretraining
    parser.add_argument('--pt_aug_lam', default=0.1, type=float)

    # --mixup_lam is the lamda hyperparameter for mixup during training
    parser.add_argument('--mixup_lam', default=0.3, type=float)

    # --train_mask_prob is the probability of masking certain features during training
    parser.add_argument('--train_mask_prob', default=0, type=float)

    # --mask_prob is the probability of masking certain features generally
    parser.add_argument('--mask_prob', default=0, type=float)

    # --ssl_avail_y is used if certain self-supervised learning labels are available
    parser.add_argument('--ssl_avail_y', default=0, type=int)

    # --pt_projhead_style sets the style of projection head used for SSL tasks
    parser.add_argument('--pt_projhead_style', default='diff', type=str,
        choices=['diff','same','nohead'])

    # --nce_temp is a temperature hyperparameter for contrastive loss
    parser.add_argument('--nce_temp', default=0.7, type=float)

    # --lam0, lam1, lam2, lam3 are weighting hyperparameters used in the pretraining tasks
    parser.add_argument('--lam0', default=0.5, type=float)
    parser.add_argument('--lam1', default=10, type=float)
    parser.add_argument('--lam2', default=1, type=float)
    parser.add_argument('--lam3', default=10, type=float)

    # --final_mlp_style sets the style of the final MLP used in the model, 'common' or 'sep'
    parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common','sep'])

    # **IMPORTANT**: We parse with an empty list so no external arguments can be passed.
    opt = parser.parse_args(args=[])

    return opt

# Construct the path to save the model; uses savemodelroot, the task, dataset ID, and run name
modelsave_path = os.path.join(os.getcwd(), opt.savemodelroot, opt.task, str(opt.dset_id), opt.run_name)

# If the task is 'regression', set opt.dtask to 'reg'; otherwise set to 'clf' for classification
if opt.task == 'regression':
    opt.dtask = 'reg'
else:
    opt.dtask = 'clf'

# Decide whether to use GPU (CUDA) or CPU based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

# Set the random seed for reproducibility
torch.manual_seed(opt.set_seed)

# Create the directory for model checkpoints if it doesn't already exist
os.makedirs(modelsave_path, exist_ok=True)

# -------------------
# OPTIONAL: WANDB LOGGING
# -------------------
if opt.active_log:
    import wandb
    # If we are doing pretraining, log it accordingly
    if opt.pretrain:
        wandb.init(project="saint_v2_all", group=opt.run_name,
                   name=f'pretrain_{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
    else:
        # For multiclass, log to a slightly different project name; else default
        if opt.task == 'multiclass':
            wandb.init(project="saint_v2_all_kamal", group=opt.run_name,
                       name=f'{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
        else:
            wandb.init(project="saint_v2_all", group=opt.run_name,
                       name=f'{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')

# Print a message to indicate dataset is being downloaded and processed
print('Downloading and processing the dataset, it might take some time.')

# -------------------
# DATA PREPARATION USING OPENML
# -------------------
# data_prep_openml returns the categorical dimensions, indexes, and continuous indexes,
# as well as the train, validation, and test splits for both data and labels
cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = \
    data_prep_openml(opt.dset_id, opt.dset_seed, opt.task, datasplit=[.65, .15, .2])

# Store the mean and std of the continuous features (for normalization) in an array
continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)


# -------------------
# HYPERPARAMETER ADJUSTMENTS BASED ON DATA
# -------------------
# nfeat is the total number of input features (columns) in X_train
_, nfeat = X_train['data'].shape

# If the dataset has > 100 features, reduce the embedding size and/or batch size for memory constraints
if nfeat > 100:
    opt.embedding_size = min(8, opt.embedding_size)
    opt.batchsize = min(64, opt.batchsize)

# If attention type is not just 'col', modify the training configuration:
# - reduce transformer depth
# - reduce attention heads
# - increase dropout
# - limit embedding size
if opt.attentiontype != 'col':
    opt.transformer_depth = 1
    opt.attention_heads = min(4, opt.attention_heads)
    opt.attention_dropout = 0.8
    opt.embedding_size = min(32, opt.embedding_size)
    opt.ff_dropout = 0.8

# Print the number of features and batchsize being used
print(nfeat, opt.batchsize)
# Print the configuration stored in opt
print(opt)

# If logging to wandb, store the entire config
if opt.active_log:
    wandb.config.update(opt)

# -------------------
# DATASET & DATALOADER INITIALIZATION
# -------------------
# Create our custom dataset objects for training, validation, and testing
train_ds = DataSetCatCon(X_train, y_train, cat_idxs, opt.dtask, continuous_mean_std)
valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, opt.dtask, continuous_mean_std)
test_ds = DataSetCatCon(X_test, y_test, cat_idxs, opt.dtask, continuous_mean_std)

# Wrap them in DataLoaders
trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,  num_workers=4)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)
testloader  = DataLoader(test_ds,  batch_size=opt.batchsize, shuffle=False, num_workers=4)

# Determine the dimension of the output (y_dim) based on the task
# For binary classification, y_dim = 2
# For multiclass classification, y_dim = number of unique classes
# For regression, y_dim = 1
if opt.task == 'regression':
    y_dim = 1
else:
    y_dim = len(np.unique(y_train['data'][:, 0]))

# cat_dims array holds the distinct number of categories for each categorical feature.
# We prepend a 1 for the [CLS] token (a special token used in transformers).
cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)

# -------------------
# MODEL INITIALIZATION
# -------------------
model = SAINT(
    categories=tuple(cat_dims),
    num_continuous=len(con_idxs),
    dim=opt.embedding_size,
    dim_out=1,  # dimension of each feature embedding's output
    depth=opt.transformer_depth,
    heads=opt.attention_heads,
    attn_dropout=opt.attention_dropout,
    ff_dropout=opt.ff_dropout,
    mlp_hidden_mults=(4, 2),
    cont_embeddings=opt.cont_embeddings,
    attentiontype=opt.attentiontype,
    final_mlp_style=opt.final_mlp_style,
    y_dim=y_dim
)

# vision_dset stores whether this is a vision dataset (controlled by --vision_dset)
vision_dset = opt.vision_dset

# -------------------
# LOSS CRITERION SETUP
# -------------------
# Depending on the task and y_dim, choose an appropriate loss function
if y_dim == 2 and opt.task == 'binary':
    # For binary classification with 2 output classes
    criterion = nn.CrossEntropyLoss().to(device)
elif y_dim > 2 and opt.task == 'multiclass':
    # For multiclass classification
    criterion = nn.CrossEntropyLoss().to(device)
elif opt.task == 'regression':
    # For regression tasks, use mean-squared error
    criterion = nn.MSELoss().to(device)
else:
    # Raise an error if none of these conditions are met
    raise 'case not written yet'

# Move the model to the chosen device (GPU if available, else CPU)
model.to(device)

# -------------------
# OPTIONAL PRETRAINING
# -------------------
if opt.pretrain:
    from pretraining import SAINT_pretrain
    # Perform pretraining (according to tasks like contrastive or denoising)
    model = SAINT_pretrain(model, cat_idxs, X_train, y_train, continuous_mean_std, opt, device)

# -------------------
# OPTIMIZER & SCHEDULER
# -------------------
if opt.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                          momentum=0.9, weight_decay=5e-4)
    from utils import get_scheduler
    scheduler = get_scheduler(opt, optimizer)
elif opt.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
elif opt.optimizer == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)

# Initialize variables to keep track of the best validation metrics
best_valid_auroc = 0
best_valid_accuracy = 0
best_test_auroc = 0
best_test_accuracy = 0
best_valid_rmse = 100000  # a large initial value for RMSE tracking

print('Training begins now.')

# -------------------
# MAIN TRAINING LOOP
# -------------------
for epoch in range(opt.epochs):
    # Switch the model to training mode
    model.train()
    
    # Variable to accumulate the total loss for the epoch
    running_loss = 0.0

    # Loop through each batch from the train DataLoader
    for i, data in enumerate(trainloader, 0):
        # Zero the gradients (PyTorch accumulates gradients)
        optimizer.zero_grad()

        # data[0] = x_categ (categorical features)
        # data[1] = x_cont (continuous features)
        # data[2] = y_gts (ground truth labels)
        # data[3] = cat_mask (mask for categorical features)
        # data[4] = con_mask (mask for continuous features)
        x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), \
                                                    data[1].to(device), \
                                                    data[2].to(device), \
                                                    data[3].to(device), \
                                                    data[4].to(device)

        # embed_data_mask converts raw data to embeddings for both categorical and continuous features
        # taking into account any masking that might be required.
        _, x_categ_enc, x_cont_enc = embed_data_mask(
            x_categ, x_cont, cat_mask, con_mask, model, vision_dset
        )

        # Pass the encoded embeddings through the Transformer
        reps = model.transformer(x_categ_enc, x_cont_enc)

        # reps[:,0,:] corresponds to the [CLS] token's representation, which is often used
        # as a summarization vector for classification or regression
        y_reps = reps[:, 0, :]

        # mlpfory is the final MLP layer that predicts the output (class or reg)
        y_outs = model.mlpfory(y_reps)

        # Calculate the loss based on the task
        if opt.task == 'regression':
            loss = criterion(y_outs, y_gts)
        else:
            # For classification, we typically have class indices in y_gts, so we use squeeze
            loss = criterion(y_outs, y_gts.squeeze())

        # Backpropagate the loss to compute gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        # If we are using SGD with a scheduler, step the scheduler once per iteration
        if opt.optimizer == 'SGD':
            scheduler.step()

        # Accumulate the loss
        running_loss += loss.item()

    # Optionally log the training loss to wandb
    if opt.active_log:
        wandb.log({
            'epoch': epoch,
            'train_epoch_loss': running_loss,
            'loss': loss.item()
        })

    # Every 5 epochs, check performance on validation and test sets
    if epoch % 5 == 0:
        # Switch the model to eval mode for validation
        model.eval()
        with torch.no_grad():
            if opt.task in ['binary', 'multiclass']:
                # classification_scores calculates accuracy and AUROC
                accuracy, auroc = classification_scores(model, validloader, device, opt.task, vision_dset)
                test_accuracy, test_auroc = classification_scores(model, testloader, device, opt.task, vision_dset)

                print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                      (epoch + 1, accuracy, auroc))
                print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                      (epoch + 1, test_accuracy, test_auroc))

                # Log validation/test metrics if using wandb
                if opt.active_log:
                    wandb.log({'valid_accuracy': accuracy, 'valid_auroc': auroc})
                    wandb.log({'test_accuracy': test_accuracy, 'test_auroc': test_auroc})

                # Save the model if it achieves better validation accuracy (and/or AUROC)
                # For multiclass, we track best accuracy.
                if opt.task == 'multiclass':
                    if accuracy > best_valid_accuracy:
                        best_valid_accuracy = accuracy
                        best_test_auroc = test_auroc
                        best_test_accuracy = test_accuracy
                        torch.save(model.state_dict(), f'{modelsave_path}/bestmodel.pth')
                else:
                    # For binary, track best accuracy as well, or best AUROC if preferred
                    if accuracy > best_valid_accuracy:
                        best_valid_accuracy = accuracy
                        best_test_auroc = test_auroc
                        best_test_accuracy = test_accuracy
                        torch.save(model.state_dict(), f'{modelsave_path}/bestmodel.pth')

            else:
                # For regression, compute RMSE on valid and test sets
                valid_rmse = mean_sq_error(model, validloader, device, vision_dset)
                test_rmse = mean_sq_error(model, testloader, device, vision_dset)

                print('[EPOCH %d] VALID RMSE: %.3f' %
                      (epoch + 1, valid_rmse))
                print('[EPOCH %d] TEST RMSE: %.3f' %
                      (epoch + 1, test_rmse))

                # Log these metrics if wandb is active
                if opt.active_log:
                    wandb.log({'valid_rmse': valid_rmse, 'test_rmse': test_rmse})

                # If this validation RMSE is better than our previous best, save the model
                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    best_test_rmse = test_rmse
                    torch.save(model.state_dict(), f'{modelsave_path}/bestmodel.pth')

        # Switch back to train mode after validation
        model.train()

# -------------------
# AFTER TRAINING COMPLETES
# -------------------
# Count total number of parameters in the model for reporting
total_parameters = count_parameters(model)
print('TOTAL NUMBER OF PARAMS: %d' % (total_parameters))

# Print out the best results on the test set depending on the task
if opt.task == 'binary':
    print('AUROC on best model:  %.3f' % (best_test_auroc))
elif opt.task == 'multiclass':
    print('Accuracy on best model:  %.3f' % (best_test_accuracy))
else:
    print('RMSE on best model:  %.3f' % (best_test_rmse))

# Also log these final results to wandb if active
if opt.active_log:
    if opt.task == 'regression':
        wandb.log({
            'total_parameters': total_parameters,
            'test_rmse_bestep': best_test_rmse,
            'cat_dims': len(cat_idxs),
            'con_dims': len(con_idxs)
        })
    else:
        wandb.log({
            'total_parameters': total_parameters,
            'test_auroc_bestep': best_test_auroc,
            'test_accuracy_bestep': best_test_accuracy,
            'cat_dims': len(cat_idxs),
            'con_dims': len(con_idxs)
        })
