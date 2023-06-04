# %%

import os
import torch
import numpy as np
wds = ["/Users/gopalpenny/Projects/ml/classy/transformer_classifier",
       "/Users/gopal/Projects/ml/classy/transformer_classifier"]
wd_exists = [x for x in wds if os.path.exists(x)][0]
os.chdir(wd_exists)
# %%

import sys
sys.path.insert(0, './transformer_classifier')

# import importlib
# importlib.reload(['train_transformer'])
from train_transformer import train_transformer_func

# %%
s1_data_path = "data/s1_data_prepped.pt"
s2_data_path = "data/s2_data_prepped.pt"
labels_path = 'data/model_data_labels.pt'
norms_path = "data/model_data_norms.pt"
output_dir_path = "./s1_s2_train"


train_transformer_func(s1_data_path, s2_data_path, norms_path, labels_path, output_dir_path)

# %%

# s1_data_path = "ubon_data/recheckingin/U_U2_S1_32_RS.pt"
# s2_data_path = "ubon_data/recheckingin/U_U2_S2_32_B8432_RSN_1122.pt"
# labels_path = "ubon_data/recheckingin/model_data_labels_new.pt"
# norms_path = "data/model_data_norms.pt"
# output_dir_path = "./s1_train"

# s1_data = torch.load(s1_data_path)
# labels = torch.load(labels_path)

# len(np.unique(labels[:,0]))

# # %%

# train_transformer_s1(s1_data_path, norms_path, labels_path, output_dir_path)

# %%
