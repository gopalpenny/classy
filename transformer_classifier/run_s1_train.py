# %%

import os

wds = ["/Users/gopalpenny/Projects/ml/classy/transformer_classifier",
       "/Users/gopal/Projects/ml/classy/transformer_classifier"]
wd_exists = [x for x in wds if os.path.exists(x)][0]
os.chdir(wd_exists)
# %%

import sys
sys.path.insert(0, './transformer_classifier')
from train_s1_transformer import train_transformer_s1

# %%
s1_data_path = "data/s1_data_prepped.pt"
labels_path = 'data/model_data_labels.pt'
norms_path = "data/model_data_norms.pt"
output_dir_path = "./s1_train"


train_transformer_s1(s1_data_path, norms_path, labels_path, output_dir_path)

# %%
