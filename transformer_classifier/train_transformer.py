# %%

import os
import torch
import numpy as np

# %%
# Set working directory
wds = ["/Users/gopalpenny/Projects/ml/classy/transformer_classifier",
       "/Users/gopal/Projects/ml/classy/transformer_classifier"]
wd_exists = [x for x in wds if os.path.exists(x)][0]
os.chdir(wd_exists)
# %%
import sys
sys.path.insert(0, './transformer_classifier')

# import importlib
# importlib.reload(['train_transformer'])
from train_transformer_function import train_transformer_func
from transformer_sentinel import TransformerClassifier
# %%
###### Set paths to input data ######
s1_data_path = "data/s1_data_prepped.pt"
s2_data_path = "data/s2_data_prepped.pt"
labels_path = 'data/model_data_labels.pt'
norms_path = "data/model_data_norms.pt"
output_dir_path = "./s1_s2_train"

# %%
# Number of training epochs
n_epochs = 25 # of training epochs

# Parameters for the transformer
# Note - dmodel must be divisible by nhead
dmodel = 36 # embedding dimention (# columns passed to attention layers)
nhead = 6 # number of heads in the multiheadattention models
dhid = 100 # dimension of the feedforward network model (after each attention layer)
nlayers = 3 # number of attention layers

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
train_transformer_func(xnn, s1_data_path, s2_data_path, norms_path, labels_path, output_dir_path, n_epochs)

# %%

###### TRAIN MODEL ON S1 DATA ONLY ######
# train_transformer_func(xnn, s1_data_path, None, norms_path, labels_path, output_dir_path, n_epochs)

# %%
###### TRAIN MODEL ON S2 DATA ONLY ######
# train_transformer_func(xnn, None, s2_data_path, norms_path, labels_path, output_dir_path, n_epochs)

# %%
