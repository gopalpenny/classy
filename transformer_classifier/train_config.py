## Configuration for train_transformer.py

###### Set paths to input data ######
s1_data_path = "data/s1_data_prepped.pt"
s2_data_path = "data/s2_data_prepped.pt"
labels_path = 'data/model_data_labels.pt'
norms_path = "data/model_data_norms.pt"
output_dir_path = "./s1_s2_train"

# %%
# Number of training epochs
n_epochs = 100 # of training epochs

# Parameters for the transformer
# Note - dmodel must be divisible by nhead
dmodel = 36 # embedding dimension (# columns passed to attention layers)
nhead = 6 # number of heads in the multiheadattention models
dhid = 100 # dimension of the feedforward network model (after each attention layer)
nlayers = 3 # number of attention layers