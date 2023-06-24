## Configuration for train_transformer.py

###### Set paths to input data ######
s1_data_path = "data/s1_data_prepped.pt"
s2_data_path = "data/s2_data_prepped.pt"
labels_path = 'data/model_data_labels.pt'
norms_path = "data/model_data_norms.pt"
output_dir_path = "./s1_s2_train"

# %%
# Set the training, validation, and test sets -- 2 options:

# 1. For 80-10-10 splits:
# train_val_test_ids = None 

# 2. Specify the loc_ids for the training, validation, and test sets as tuple of lists
train_val_test_ids = (list(range(0,100)), # training set loc_ids
                      list(range(100,150)), # validation set loc_ids
                      list(range(100,150))) # test set (can be same or different as validation set)

# %%
# Number of training epochs
n_epochs = 50 # of training epochs
batch_size = 20 # batch size

# Parameters for the transformer
# Note - dmodel must be divisible by nhead
dmodel = 36 # embedding dimension (# columns passed to attention layers)
nhead = 6 # number of heads in the multiheadattention models
dhid = 100 # dimension of the feedforward network model (after each attention layer)
nlayers = 3 # number of attention layers
lr = 0.0005 # learning rate
weight_decay = 1e-5 # weight decay