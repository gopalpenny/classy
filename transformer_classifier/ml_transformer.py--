#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 21:35:24 2023

@author: gopal
"""

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
from torch import nn, Tensor


# %%
class SentinelDatasets(Dataset):
    """Sentinel 2 dataset"""
    
    def __init__(self, s1, s2, y, max_obs_s1, max_obs_s2):
        """
        Args:
            s1 (tensor): contains loc_id and predictors as columns, s1 observations as rows
            s2 (tensor): contains loc_id and predictors as columns, s2 observations as rows
            y (tensor): contains loc_id as rows (& first column), class as 1-hot columns
        """
        self.s1 = s1
        self.s2 = s2
        self.y = y
        self.max_obs_s1 = max_obs_s1
        self.max_obs_s2 = max_obs_s2
        # self.proj_path = proj_path
        # proj_normpath = os.path.normpath(proj_path)
        # proj_dirname = proj_normpath.split(os.sep)[-1]
        # self.proj_name = re.sub("_classification$","",proj_dirname)
        # self.class_path = os.path.join(proj_path, self.proj_name + "_classification")
        # self.ts_path = os.path.join(proj_path, self.proj_name + "_download_timeseries")
        # self.pt_classes = pd.read_csv(os.path.join(self.class_path,"location_classification.csv"))
        # self.pt_classes = classes[['loc_id', class_colname]].dropna()
        # self.classes = pd.unique(self.pt_classes[class_colname])
        # self.labels = self.pt_classes.assign(val = 1).pivot_table(columns = class_colname, index = 'loc_id', values = 'val', fill_value= 0)

    
    def __getitem__(self, idx):
        # get loc_id
        loc_id = self.y[idx,0]
        self.last_loc_id = loc_id
        
        # select location id
        s1_loc = self.s1[self.s1[:,0]==loc_id]
        s1_prep = s1_loc[:,1:] # remove loc_id column
        
        # pad zeros to max_obs
        n_pad_s1 = self.max_obs_s1 - s1_prep.shape[0]
        
        s1 = torch.cat((s1_prep, torch.zeros(n_pad_s1, s1_prep.shape[1])), dim = 0)
        
        s1 = s1.float()
        
        
        # select location id
        s2_loc = self.s2[self.s2[:,0]==loc_id]
        s2_prep = s2_loc[:,1:] # remove loc_id column
        
        # pad zeros to max_obs
        n_pad_s2 = self.max_obs_s2 - s2_prep.shape[0]
        
        s2 = torch.cat((s2_prep, torch.zeros(n_pad_s2, s2_prep.shape[1])), dim = 0)
        
        s2 = s2.float()
        
        
        # get one-hot encoding for the point as tensor
        y = self.y.clone().detach()[idx,1:].float()
        
        return s1, s2, y
        
    def __len__(self):
        return self.y.shape[0]
    
    
class TransformerClassifier(nn.Module):
    def __init__(self, ntoken: int, dmodel: int, nhead: int, dhid: int, 
                 nlayers: int, data_dim: int, nclasses: int):
        """
        data_dim: dimension of data (i.e., num of columns) including position as first dimension (but not loc_id)
        """
        super().__init__()
        self.positional_layer = nn.Linear(1, dmodel)
        self.embed_layer = nn.Linear(data_dim - 1, dmodel) # transform data to embed dimension (dmodel)
        
        # dim_feedforward: https://stackoverflow.com/questions/68087780/pytorch-transformer-argument-dim-feedforward
        # shortly: dim_feedforward is a hidden layer between two forward layers at the end of the encoder layer, passed for each word one-by-one
        self.encoderlayer = nn.TransformerEncoderLayer(d_model = dmodel, nhead = nhead, dim_feedforward = dhid)
        self.encoder = nn.TransformerEncoder(self.encoderlayer, nlayers)
        
        self.num_params = ntoken * dmodel
        
        self.class_encoder = nn.Linear(dmodel, nclasses)
    
    def forward(self, src: Tensor) -> Tensor:
        
        positions = src[:, :, 0:1]
        data = src[:, :, 1:]
        pe = self.positional_layer(positions)
        data_embed = self.embed_layer(data)
        data_and_pe = pe + data_embed
        encoder_out = self.encoder(data_and_pe)
        
        maxpool = torch.max(encoder_out,dim = 1)[0]
        
        # softmax ensures output of model is probability of class membership -- which sum to 1
        # BUT this is already done with CrossEntropyLoss so it's not necessary for this loss function
        classes_one_hot = self.class_encoder(maxpool) #, dim = 1
        
        
        classes = classes_one_hot #torch.softmax(classes_one_hot, 0)
        
        # classes = nn.functional.softmax(classes, 1) # don't use softmax with cross entropy loss... or do?
        # don't: https://stackoverflow.com/questions/55675345/should-i-use-softmax-as-output-when-using-cross-entropy-loss-in-pytorch
        # do: Machine Learning with Pytorch and Scikitlearn (p 471: Loss functions for classifiers) -- BUT NOT WITH CROSS ENTROPY LOSS (p478
        
        return classes

        # data_in = tf_test[:, :, 1:] # select only the data
        # positions = tf_test[:,:,0:1] # split out positional data
        # data_dim = data_in.shape[-1]

# class s2Dataset(Dataset):
#     """Sentinel 2 dataset"""
    
#     def __init__(self, x, y, max_obs):
#         """
#         Args:
#             x (tensor): contains loc_id and predictors as columns, s2 observations as rows
#             y (tensor): contains loc_id as rows (& first column), class as 1-hot columns
#         """
#         self.x = x
#         self.y = y
#         self.max_obs = max_obs
#         # self.proj_path = proj_path
#         # proj_normpath = os.path.normpath(proj_path)
#         # proj_dirname = proj_normpath.split(os.sep)[-1]
#         # self.proj_name = re.sub("_classification$","",proj_dirname)
#         # self.class_path = os.path.join(proj_path, self.proj_name + "_classification")
#         # self.ts_path = os.path.join(proj_path, self.proj_name + "_download_timeseries")
#         # self.pt_classes = pd.read_csv(os.path.join(self.class_path,"location_classification.csv"))
#         # self.pt_classes = classes[['loc_id', class_colname]].dropna()
#         # self.classes = pd.unique(self.pt_classes[class_colname])
#         # self.labels = self.pt_classes.assign(val = 1).pivot_table(columns = class_colname, index = 'loc_id', values = 'val', fill_value= 0)

    
#     def __getitem__(self, idx):
#         # get loc_id
#         loc_id = self.y[idx,0]
#         self.last_loc_id = loc_id
        
#         # select location id
#         x_loc = self.x[self.x[:,0]==loc_id]
#         x_prep = x_loc[:,1:] # remove loc_id column
        
#         # pad zeros to max_obs
#         n_pad = self.max_obs - x_prep.shape[0]
        
#         x = torch.cat((x_prep, torch.zeros(n_pad, x_prep.shape[1])), dim = 0)
        
#         x = x.float()
        
        
        
#         # get one-hot encoding for the point as tensor
#         y = torch.tensor(self.y[idx,1:]).float().flatten()
        
#         return x, y
        
#     def __len__(self):
#         return self.y.shape[0]