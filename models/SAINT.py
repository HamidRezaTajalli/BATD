import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
import numpy as np
from saint_lib.prepare import argparse_prepare



class SAINTModel:

    def __init__(self, data_obj, is_numerical=False):

        if data_obj.num_classes == 2:
            task = 'binary'
        elif data_obj.num_classes > 2:
            task = 'multiclass'
        else:
            raise ValueError(f"Invalid number of classes for data object: {data_obj.num_classes}")
        
        self.opt = argparse_prepare()
        self.opt.task = task

        # If the task is 'regression', set opt.dtask to 'reg'; otherwise set to 'clf' for classification
        if self.opt.task == 'regression':
            self.opt.dtask = 'reg'
        else:
            self.opt.dtask = 'clf'

        
        # -------------------
        # HYPERPARAMETER ADJUSTMENTS BASED ON DATA
        # -------------------
        # nfeat is the total number of input features (columns) in X_train
        nfeat = len(data_obj.feature_names)

        # If the dataset has > 100 features, reduce the embedding size and/or batch size for memory constraints
        if nfeat > 100:
            self.opt.embedding_size = min(8, self.opt.embedding_size)
            self.opt.batchsize = min(64, self.opt.batchsize)

        
        # If attention type is not just 'col', modify the training configuration:
        # - reduce transformer depth
        # - reduce attention heads
        # - increase dropout
        # - limit embedding size
        if self.opt.attentiontype != 'col':
            self.opt.transformer_depth = 1
            self.opt.attention_heads = min(4, self.opt.attention_heads)
            self.opt.attention_dropout = 0.8
            self.opt.embedding_size = min(32, self.opt.embedding_size)
            self.opt.ff_dropout = 0.8

        # Print the number of features and batchsize being used
        print(nfeat, self.opt.batchsize)
        # Print the configuration stored in opt
        print(self.opt)

        # # If logging to wandb, store the entire config
        # if self.opt.active_log:
        #     wandb.config.update(self.opt)

        if is_numerical or not data_obj.cat_cols:
            self.secat_dims, self.secat_idxs = [], []
            self.con_idxs = data_obj.num_cols_idx
        else:
            self.secat_dims = data_obj.FTT_n_categories
            self.secat_idxs = data_obj.cat_cols_idx
            self.con_idxs = data_obj.num_cols_idx


        