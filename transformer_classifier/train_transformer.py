# %%

import os
import torch
import numpy as np

# %%
# Set working directory
wds = ["/Users/gopalpenny/Projects/ml/classy/transformer_classifier",
       "/Users/gopal/Projects/ml/classy/transformer_classifier",
       "/home/svu/gpenny/Projects/classy/transformer_classifier"]
wd_exists = [x for x in wds if os.path.exists(x)][0]
os.chdir(wd_exists)
# %%
import sys
sys.path.insert(0, './transformer_classifier')

# import importlib
# importlib.reload(['train_transformer'])
from train_transformer_function import train_transformer_func
from transformer_sentinel import TransformerClassifier
import train_config as cfg
# %%
###### Set paths to input data ######
s1_data_path = cfg.s1_data_path
s2_data_path = cfg.s2_data_path
labels_path = cfg.labels_path
norms_path = cfg.norms_path
output_dir_path = cfg.output_dir_path

# %%
train_val_test_ids = cfg.train_val_test_ids

# %%
# Number of training epochs and batch size
n_epochs = cfg.n_epochs # of training epochs
batch_size = cfg.batch_size # batch size

# Parameters for the transformer
# Note - dmodel must be divisible by nhead
dmodel = cfg.dmodel # embedding dimention (# columns passed to attention layers)
nhead = cfg.nhead # number of heads in the multiheadattention models
dhid = cfg.dhid # dimension of the feedforward network model (after each attention layer)
nlayers = cfg.nlayers # number of attention layers
lr = cfg.lr # learning rate
weight_decay = cfg.weight_decay # weight decay

# These variables shouldn't change
s1_dim = 4 # number of columns in s1 and s2 data (excluding loc_id)
s2_dim = 5 # number of columns in s1 and s2 data (excluding loc_id)
nclasses = 4 # number of classes in the data


# %%
xnn = TransformerClassifier(dmodel = dmodel, # embedding dimention (# columns passed to attention layers)
                            nhead = nhead, # number of heads in the multiheadattention models
                            dhid = dhid, # dimension of the feedforward network model (after each attention layer)
                            nlayers = nlayers, # number of attention layers
                            s1_dim = s1_dim, s2_dim = s2_dim, # number of columns in s1 and s2 data (excluding loc_id)
                            nclasses = nclasses) # number of classes in the data

# %%
###### TRAIN MODEL ON S1 AND S2 DATA ######
train_transformer_func(xnn, s1_data_path, s2_data_path, norms_path, labels_path, output_dir_path, n_epochs, batch_size, lr, weight_decay, train_val_test_ids)

# %%

###### TRAIN MODEL ON S1 DATA ONLY ######
# train_transformer_func(xnn, s1_data_path, None, norms_path, labels_path, output_dir_path, n_epochs, batch_size, lr, weight_decay)

# %%
###### TRAIN MODEL ON S2 DATA ONLY ######
# train_transformer_func(xnn, None, s2_data_path, norms_path, labels_path, output_dir_path, n_epochs, batch_size, lr, weight_decay)

# %%
