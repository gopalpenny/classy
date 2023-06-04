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
# class SentinelDatasets(Dataset):
#     """Sentinel 2 dataset"""
    
#     def __init__(self, s1, s2, y, max_obs_s1, max_obs_s2):
#         """
#         Args:
#             s1 (tensor): contains loc_id and predictors as columns, s1 observations as rows
#             s2 (tensor): contains loc_id and predictors as columns, s2 observations as rows
#             y (tensor): contains loc_id as rows (& first column), class as 1-hot columns
#         """
#         self.s1 = s1
#         self.s2 = s2
#         self.y = y
#         self.max_obs_s1 = max_obs_s1
#         self.max_obs_s2 = max_obs_s2
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
#         s1_loc = self.s1[self.s1[:,0]==loc_id]
#         s1_prep = s1_loc[:,1:] # remove loc_id column
        
#         # pad zeros to max_obs
#         n_pad_s1 = self.max_obs_s1 - s1_prep.shape[0]
        
#         s1 = torch.cat((s1_prep, torch.zeros(n_pad_s1, s1_prep.shape[1])), dim = 0)
        
#         s1 = s1.float()
        
        
#         # select location id
#         s2_loc = self.s2[self.s2[:,0]==loc_id]
#         s2_prep = s2_loc[:,1:] # remove loc_id column
        
#         # pad zeros to max_obs
#         n_pad_s2 = self.max_obs_s2 - s2_prep.shape[0]
        
#         s2 = torch.cat((s2_prep, torch.zeros(n_pad_s2, s2_prep.shape[1])), dim = 0)
        
#         s2 = s2.float()
        
        
#         # get one-hot encoding for the point as tensor
#         y = self.y.clone().detach()[idx,1:].float()
        
#         return s1, s2, y, loc_id
        
#     def __len__(self):
#         return self.y.shape[0]

class SentinelDataModule(nn.Module):
    def __init__(self, data_dim, dmodel):
        super().__init__()
        self.positional_layer = nn.Linear(1, dmodel)
        self.embed_layer = nn.Linear(data_dim - 1, dmodel)
    
    def forward(self, src: Tensor) -> Tensor:
        positions = src[:, :, 0:1]
        data = src[:, :, 1:]
        pe = self.positional_layer(positions)
        data_embed = self.embed_layer(data)
        data_and_pe = pe + data_embed
        return data_and_pe
    
class TransformerClassifier(nn.Module):
    def __init__(self, dmodel: int, nhead: int, dhid: int, 
                 nlayers: int, s1_dim: int, s2_dim: int, nclasses: int):
        """
        ntoken: number of tokens
        dmodel: the number of features (columns) in the transformer input
        nhead: number of heads in the multiheadattention model
        dhid: dimension of the feedforward network model
        nlayers: number of encoder layers in the transformer model
        s1_dim: dimension of s1 data (i.e., num of columns) including position as first dimension (but not loc_id)
        s2_dim: dimension of s2 data (i.e., num of columns) including position as first dimension (but not loc_id)
        nclasses: number of classes to calculate probabilities for

        Can set s1_dim to zero, or s2_dim to zero and it will only use the other data
        """
        super().__init__()

        # for positional and data embedding
        self.s1nn = SentinelDataModule(data_dim = s1_dim, dmodel = dmodel)
        self.s2nn = SentinelDataModule(data_dim = s2_dim, dmodel = dmodel)
        self.dmodel = dmodel
        self.s1_dim = s1_dim
        self.s2_dim = s2_dim
        
        # dim_feedforward: https://stackoverflow.com/questions/68087780/pytorch-transformer-argument-dim-feedforward
        # shortly: dim_feedforward is a hidden layer between two forward layers at the end of the encoder layer, passed for each word one-by-one
        self.encoderlayer = nn.TransformerEncoderLayer(d_model = self.dmodel, nhead = nhead, dim_feedforward = dhid)
        self.encoder = nn.TransformerEncoder(self.encoderlayer, nlayers)
        
        # self.num_params = ntoken * self.dmodel
        
        self.class_encoder = nn.Linear(self.dmodel, nclasses)
    
    def forward(self, s1: Tensor, s2: Tensor) -> Tensor:
        
        if self.s1_dim > 0:
            s1_data_and_pe = self.s1nn(s1)
        
        if self.s2_dim > 0:
            s2_data_and_pe = self.s2nn(s2)

        if self.s1_dim > 0 and self.s2_dim == 0:
            data_and_pe = s1_data_and_pe
        elif self.s1_dim == 0 and self.s2_dim > 0:
            data_and_pe = s2_data_and_pe
        else:
            data_and_pe = torch.cat((s1_data_and_pe, s2_data_and_pe), dim = 1)
        
        encoder_out = self.encoder(data_and_pe)
        
        maxpool = torch.max(encoder_out, dim = 1)[0]
        
        # softmax ensures output of model is probability of class membership -- which sum to 1
        # BUT this is already done with CrossEntropyLoss so it's not necessary for this loss function
        class_likelihood = self.class_encoder(maxpool) #, dim = 1
        
        classes = class_likelihood #torch.softmax(classes_one_hot, 0)
        
        # classes = nn.functional.softmax(classes, 1) # don't use softmax with cross entropy loss... or do?
        # don't: https://stackoverflow.com/questions/55675345/should-i-use-softmax-as-output-when-using-cross-entropy-loss-in-pytorch
        # do: Machine Learning with Pytorch and Scikitlearn (p 471: Loss functions for classifiers) -- BUT NOT WITH CROSS ENTROPY LOSS (p478
        
        return classes
    
# class TransformerClassifierS1(nn.Module):
#     def __init__(self, ntoken: int, dmodel: int, nhead: int, dhid: int, 
#                  nlayers: int, data_dim: int, nclasses: int):
#         """
#         data_dim: dimension of data (i.e., num of columns) including position as first dimension (but not loc_id)
#         """
#         super().__init__()
#         self.positional_layer = nn.Linear(1, dmodel)
#         self.embed_layer = nn.Linear(data_dim - 1, dmodel) # transform data to embed dimension (dmodel)
        
#         # dim_feedforward: https://stackoverflow.com/questions/68087780/pytorch-transformer-argument-dim-feedforward
#         # shortly: dim_feedforward is a hidden layer between two forward layers at the end of the encoder layer, passed for each word one-by-one
#         self.encoderlayer = nn.TransformerEncoderLayer(d_model = dmodel, nhead = nhead, dim_feedforward = dhid)
#         self.encoder = nn.TransformerEncoder(self.encoderlayer, nlayers)
        
#         self.num_params = ntoken * dmodel
        
#         self.class_encoder = nn.Linear(dmodel, nclasses)
    
#     def forward(self, src: Tensor) -> Tensor:
        
#         positions = src[:, :, 0:1]
#         data = src[:, :, 1:]
#         pe = self.positional_layer(positions)
#         data_embed = self.embed_layer(data)
#         data_and_pe = pe + data_embed
#         encoder_out = self.encoder(data_and_pe)
        
#         maxpool = torch.max(encoder_out,dim = 1)[0]
        
#         # softmax ensures output of model is probability of class membership -- which sum to 1
#         # BUT this is already done with CrossEntropyLoss so it's not necessary for this loss function
#         classes_one_hot = self.class_encoder(maxpool) #, dim = 1
        
        
#         classes = classes_one_hot #torch.softmax(classes_one_hot, 0)
        
#         # classes = nn.functional.softmax(classes, 1) # don't use softmax with cross entropy loss... or do?
#         # don't: https://stackoverflow.com/questions/55675345/should-i-use-softmax-as-output-when-using-cross-entropy-loss-in-pytorch
#         # do: Machine Learning with Pytorch and Scikitlearn (p 471: Loss functions for classifiers) -- BUT NOT WITH CROSS ENTROPY LOSS (p478
        
#         return classes

#         # data_in = tf_test[:, :, 1:] # select only the data
#         # positions = tf_test[:,:,0:1] # split out positional data
#         # data_dim = data_in.shape[-1]
        
# %%
def getitem_sentinel_data(s_data, loc_id, max_obs_s, resample_days, resample_days_n):

    # select location id
    s_loc = s_data[s_data[:,0]==loc_id]
    s_prep = s_loc[:,1:] # remove loc_id column

    # resample days if max_obs_s is less than number of observations
    if max_obs_s < s_prep.shape[0] and resample_days:

        days_select = torch.arange(0, 370, resample_days_n)
        s_prep = resample_id_nearest_days(tensor_full = s_prep, 
                                                days_select = days_select, 
                                                id_col = 0, 
                                                day_col = 1)
    
    # pad zeros to max_obs and ensure float
    n_pad_s = max_obs_s - s_prep.shape[0]
    s = torch.cat((s_prep, torch.zeros(n_pad_s, s_prep.shape[1])), dim = 0)
    s = s.float()


class SentinelDataset(Dataset):
    """Sentinel 1 & 2 dataset"""
    
    def __init__(self, y, s1 = None, s2 = None, max_obs_s1 = None, max_obs_s2 = None, resample_days = False, resample_days_n = 0):
        """
        Args:
            s1 (tensor): contains loc_id and predictors as columns, s1 observations as rows
            s2 (tensor): contains loc_id and predictors as columns, s2 observations as rows
            y (tensor): contains loc_id as rows (& first column), class as 1-hot columns
            max_obs_s1: maximum number of observations per location
            max_obs_s2: maximum number of observations per location
            resample_days: if True, resample to day increment given by resample_days_n
            resample_days_n: The day increment to resample to. If 0, then resample_days_n = ceil(366 / max_obs_sx)
        """
        self.s1 = s1
        self.s2 = s2
        self.y = y
        self.max_obs_s1 = max_obs_s1
        self.max_obs_s2 = max_obs_s2
        self.resample_days = resample_days
        self.resample_days_n = resample_days_n
        if resample_days and resample_days_n == 0:
            self.resample_days_n = torch.ceil(366 / max_obs_s1)

    
    def __getitem__(self, idx):
        # get loc_id
        loc_id = self.y[idx,0]
        self.last_loc_id = loc_id

        # get class for the point as tensor
        y = self.y.clone().detach()[idx,1:].float()

        if self.s1 is not None:
            # select location id and remove loc_id column
            s1 = self.s1[self.s1[:,0]==loc_id, 1:]
            s1 = s1.float()
            # s1 = getitem_sentinel_data(s_data = self.s1, loc_id = loc_id, max_obs_s = self.max_obs_s1,
            #                            resample_days = self.resample_days, resample_days_n = self.resample_days_n)
        
        if self.s2 is not None:
            # select location id and remove loc_id column
            s2 = self.s2[self.s2[:,0]==loc_id, 1:] 
            s2 = s2.float()
            # s2 = getitem_sentinel_data(s_datssa = self.s2, loc_id = loc_id, max_obs_s = self.max_obs_s2,
            #                            resample_days = self.resample_days, resample_days_n = self.resample_days_n)
        
        if ((self.s1 is not None) and (self.s2 is None)):
            return s1, y, loc_id
        elif ((self.s1 is None) and (self.s2 is not None)):
            return s2, y, loc_id
        else:
            return s1, s2, y, loc_id
        
    def __len__(self):
        return self.y.shape[0]
    
# # %%
# class S1Dataset(Dataset):
#     """Sentinel 1 dataset"""
    
#     def __init__(self, s1, y, max_obs_s1, resample_days = False, resample_days_n = 0):
#         """
#         Args:
#             s1 (tensor): contains loc_id and predictors as columns, s1 observations as rows
#             y (tensor): contains loc_id as rows (& first column), class as 1-hot columns
#             max_obs_s1: maximum number of observations per location
#             resample_days: if True, resample to day increment given by resample_days_n
#             resample_days_n: The day increment to resample to. If 0, then resample_days_n = ceil(366 / max_obs_s1)
#         """
#         self.s1 = s1
#         self.y = y
#         self.max_obs_s1 = max_obs_s1
#         self.resample_days = resample_days
#         if resample_days and resample_days_n == 0:
#             self.resample_days_n = torch.ceil(366 / max_obs_s1)

    
#     def __getitem__(self, idx):
#         # get loc_id
#         loc_id = self.y[idx,0]
#         self.last_loc_id = loc_id
        
#         # select location id
#         s1_loc = self.s1[self.s1[:,0]==loc_id]
#         s1_prep = s1_loc[:,1:] # remove loc_id column

#         if self.max_obs_s1 < s1_prep.shape[0] and self.resample_days:

#             days_select = torch.arange(0, 370, self.resample_days_n)
#             s1_prep = resample_id_nearest_days(tensor_full = s1_prep, 
#                                                     days_select = days_select, 
#                                                     id_col = 0, 
#                                                     day_col = 1)
        
#         # pad zeros to max_obs and ensure float
#         n_pad_s1 = self.max_obs_s1 - s1_prep.shape[0]
#         s1 = torch.cat((s1_prep, torch.zeros(n_pad_s1, s1_prep.shape[1])), dim = 0)
#         s1 = s1.float()
        
#         # get class for the point as tensor
#         y = self.y.clone().detach()[idx,1:].float()
        
#         return s1, y, loc_id
        
#     def __len__(self):
#         return self.y.shape[0]
    

# %%
data_path = './data/model_data_norms.pt'

# %%
def scale_model_data(data_path, norms_path, data_name):
    """
    Function used to import data and scale it using norms
    
    data_path : str
        - Path to the dataset
    norms_path : str
        - path to the norms
    data_name : str
        - Can be "s1" or "s2"
    """
    
    # get column names for standard deviation and means
    if data_name == 's1':
        sd_col = 's1_col_std'
        means_col = 's1_col_means'
    elif data_name == 's2':
        sd_col = 's2_col_std'
        means_col = 's2_col_means'
    else:
        raise Exception('data_name must be "s1" or "s2"')
    
    # load original data and normalization constants
    data_orig = torch.load(data_path)
    model_norms = torch.load(norms_path)

    # scale data
    norms_std = model_norms[sd_col].unsqueeze(0).repeat(data_orig.shape[0],1)
    norms_means = model_norms[means_col].unsqueeze(0).repeat(data_orig.shape[0],1)

    data_scaled = (data_orig - norms_means)/norms_std
        
    return data_scaled  

# %%
def resample_nearest_days(tensor_orig, days_select, day_col):
    """
    Select rows from tensor orig which are nearest to at least one of the values in days_select
    days_select : tensor
        Vector of evenly-spaced days used to select rows from tensor_orig
    tensor_orig: tensor
        2D tensor with 1 column being the time variable (i.e., days)
    day_col : numeric
        Colum index of tensor_orig containing time variable (days)
    """
    days = tensor_orig[:, day_col]
    
    # tensor_orig
    days_mat = torch.unsqueeze(days, 0).repeat(len(days_select), 1) #.shape
    select_mat = days_select.unsqueeze(1).repeat(1, len(days)) #.shape

    # days_mat #- select_mat
    nearest = torch.argmin(torch.abs(days_mat - select_mat), dim = 1)
    # torch.unsqueeze(torch.from_numpy(days_select),1)
    tensor_resampled = tensor_orig[torch.unique(nearest),:]
    
    return tensor_resampled
    
def resample_id_nearest_days(tensor_full, days_select, id_col, day_col):
    """
    For each id in id_col, use resample_nearest_days to resample days to the closest to days_select

    # Example:

    s1_tensor = torch.zeros(, 2)
    days_select = torch.arange(0, 370, 6)
    s1_ts_resampled = resample_id_nearest_days(tensor_full = s1_tensor, 
                                            days_select = days_select, 
                                            id_col = 0, 
                                            day_col = 1)
    """

    if tensor_full.type() == "torch.HalfTensor":
        halfTensor = True
        tensor_full = tensor_full.float()
    else:
        halfTensor = False
    ts_resampled = torch.zeros(0, tensor_full.shape[1])
    for loc_id in torch.unique(tensor_full[:, id_col]):
        # print(loc_id)
        tensor_orig = tensor_full[tensor_full[:, id_col] == loc_id]
        
        loc_resampled = resample_nearest_days(tensor_orig, days_select, day_col = 1)
        ts_resampled = torch.concat((ts_resampled, loc_resampled), dim = 0)#.shape
    
    if halfTensor:
        ts_resampled = ts_resampled.half()

    return ts_resampled

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