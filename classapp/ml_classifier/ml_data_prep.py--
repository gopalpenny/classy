#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 19:09:40 2023

@author: gopal
"""

import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import re
import sys
from datetime import timedelta
# from torch.nn.functional import normalize

# %%

proj_paths = ["/Users/gopal/Google Drive/_Research/Research projects/ML/manclassify/app_data/Thailand",
              "/Users/gopalpenny/Library/CloudStorage/GoogleDrive-gopalpenny@gmail.com/My Drive/_Research/Research projects/ML/manclassify/app_data/Thailand"]

proj_path = [path for path in proj_paths if os.path.exists(path)][0]

class_colname = 'Subclass2019'

# ## Prep project path
proj_normpath = os.path.normpath(proj_path)
proj_dirname = proj_normpath.split(os.sep)[-1]
proj_name = re.sub("_classification$","",proj_dirname)
class_path = os.path.join(proj_path, proj_name + "_classification")
ts_path = os.path.join(proj_path, proj_name + "_download_timeseries")

# ## Read point classes data frame, drop unused columns
pt_classes = pd.read_csv(os.path.join(class_path, "location_classification.csv"))
pt_classes = pt_classes[['loc_id', 'Class', class_colname]].dropna()

# %%

# ## Generate the torch tensor dataset
# 

# %%
# ### Define function to read timeseries: prep_s2_loc
# 
# * Read timeseries
# * Filter timeseries to date range (+/- 60 days)
# * Remove observations with clouds
# * Take the mean value for each day (occurs when multiple overpasses happen on the same day)


# prep dataset
date_range = pd.to_datetime(['2019-06-01','2020-05-31'])

def prep_s2_loc(loc_id, date_range, proj_path):
    ts_path = os.path.join(proj_path,"Thailand_download_timeseries")
    s2_csv_name = f"pt_ts_loc{loc_id}_s2.csv"
    s2_csv_path = os.path.join(ts_path, s2_csv_name)
    s2_ts = pd.read_csv(s2_csv_path)

    # extract dates from image ids
    s2_ts['datestr'] = [re.sub("(^[0-9]+)[a-zA-Z].*","\\1",x) for x in s2_ts.image_id]
    s2_ts['date'] = pd.to_datetime(s2_ts.datestr, format = "%Y%m%d")

    # subset to cloud-free days AND within date_range
    s2_ts = s2_ts[(s2_ts.date >= date_range[0] - timedelta(days = 60)) & 
                  (s2_ts.date <= date_range[1] + timedelta(days = 60)) & 
                  (s2_ts.cloudmask == 0)]

    # calculate day from startday
    date_diff = (s2_ts.date - date_range[0])
    s2_ts['day'] = [x.days for x in date_diff]
    s2_ts['loc_id'] = loc_id

    # select only predictor and position columns, return tensor
    s2_ts_x = s2_ts[['loc_id','day','B8','B4','B3','B2']]
    return s2_ts_x


# %%
def prep_s1_loc(loc_id, date_range, proj_path):
    ts_path = os.path.join(proj_path,"Thailand_download_timeseries")
    
    s1_csv_name = f"pt_ts_loc{loc_id}_s1.csv"
    s1_csv_path = os.path.join(ts_path, s1_csv_name)
    s1_ts = pd.read_csv(s1_csv_path)
    
    # extract dates from image ids
    s1_ts['datestr'] = [re.sub(".*_.*_.*_.*_([0-9]+)T[0-9]+_.*","\\1",x) for x in s1_ts.image_id]
    s1_ts['date'] = pd.to_datetime(s1_ts.datestr, format = "%Y%m%d")
        
    # subset to cloud-free days AND within date_range
    s1_ts = s1_ts[(s1_ts.date >= date_range[0] - timedelta(days = 60)) & 
                  (s1_ts.date <= date_range[1] + timedelta(days = 60))]
    
    s1_ts = s1_ts[['date','HH','VV','VH','HV','angle']]
    
    # calculate day from startday
    date_diff = (s1_ts.date - date_range[0])
    s1_ts['day'] = [x.days for x in date_diff]
    s1_ts['loc_id'] = loc_id
    
    # select only predictor and position columns, return tensor
    s1_ts_x = s1_ts[['loc_id','day','HH','VV','VH','HV','angle']]
    
    return s1_ts_x

# %%

# prep_s1_loc(1, date_range, proj_path)
# %%

# ### Get the torch tensor dataset (prep and save OR read)

# from ipywidgets import IntProgress
# from IPython.display import display

if os.path.exists(os.path.join(proj_path, 's1_ts_prepped.pt')):
    loc_s1_ts_tensor = torch.load(os.path.join(proj_path, 's1_ts_prepped.pt'))
    
else:
    # f = IntProgress(min=0, max=pt_classes.shape[0]) # instantiate the bar
    # display(f) # display the bar
    print("prepping s1 tensor file")
    
    s1_ts_list = []
    loc_id_list = []
    for i in np.arange(pt_classes.shape[0]):
        print(".")
        # loc_id = 499
        # print(loc_id)
        loc_id = pt_classes.loc_id.iloc[i]
        
        s1_ts_loc = prep_s1_loc(loc_id, date_range, proj_path)
        s1_ts_loc = s1_ts_loc.groupby(['loc_id','day'],as_index = False).mean()
        s1_ts_tor = torch.tensor(s1_ts_loc.to_numpy())
        s1_ts_list.append(s1_ts_tor)
        
    loc_s1_ts_tensor = torch.cat(s1_ts_list)
    torch.save(loc_s1_ts_tensor, os.path.join(proj_path, 's1_ts_prepped.pt'))
    
if os.path.exists(os.path.join(proj_path, 's2_ts_prepped.pt')):
    loc_s2_ts_tensor = torch.load(os.path.join(proj_path, 's2_ts_prepped.pt'))
    
else:
    print("prepping s1 tensor file")
    s2_ts_list = []
    loc_id_list = []
    for i in np.arange(pt_classes.shape[0]):
        # loc_id = 499
        print(".")
        loc_id = pt_classes.loc_id.iloc[i]
        # loc_id_list.append(loc_id)
        s2_ts_loc = prep_s2_loc(loc_id, date_range, proj_path)
        s2_ts_loc = s2_ts_loc.groupby(['loc_id','day'],as_index = False).mean()
        s2_ts_tor = torch.tensor(s2_ts_loc.to_numpy())
        s2_ts_list.append(s2_ts_tor)
        
    loc_s2_ts_tensor = torch.cat(s2_ts_list)
    torch.save(loc_s2_ts_tensor, os.path.join(proj_path, 's2_ts_prepped.pt'))

sys.getsizeof(loc_s2_ts_tensor)
sys.getsizeof(loc_s1_ts_tensor)


# %%
# ### Prep the dataset tensors
# 
# * Subset to training classes (crops & plantations)
# * Check max number of rows
# * Normalize & center
# * Split loc_id into training and test datasets

# In[30]:


# Create a merged class column where "Other" is used for nonfarm classes
pt_classes['class'] = ['Other' if x!='Farm' else y for x,y in zip(pt_classes['Class'],pt_classes['Subclass2019'])]
pt_classes


# In[45]:


print('All classes')
print(pt_classes.groupby(['Class','Subclass2019','class']).count())

train_classes = ['Crop(Double)','Crop(Single)','Plantation', 'Other']
pt_classes_ag = pt_classes[pt_classes['class'].isin(train_classes)][['class','loc_id']]
print('\nTraining dataset (pt_classes_ag)\n',pt_classes_ag)


# In[90]:
loc_train = pt_classes_ag.groupby('class', group_keys = False).apply(lambda x: x.sample(frac = 0.8))
loc_nontrain = pt_classes_ag[~pt_classes_ag['loc_id'].isin(loc_train.loc_id)]

loc_valid = loc_nontrain.groupby('class', group_keys = False).apply(lambda x: x.sample(frac = 0.5))
loc_test = loc_nontrain[~loc_nontrain['loc_id'].isin(loc_valid.loc_id)]

print('Training (loc_train summary)\n', loc_train.groupby('class').count())
print('\nValidate (loc_test summary)\n', loc_valid.groupby('class').count())
print('\nTesting (loc_test summary)\n', loc_test.groupby('class').count())

# In[81]:


# In[89]:

# %% Normalize s1 tensor
loc_s1_ts_tensor = loc_s1_ts_tensor[(loc_s1_ts_tensor[:,1] >= 0) & (loc_s1_ts_tensor[:,1] <= 365)]

loc_s1_ts_tensor[torch.isnan(loc_s1_ts_tensor)] = 0

col_means= loc_s1_ts_tensor.mean(dim = 0)#.shape #.unsqueeze(0).repeat(5,1)
col_std= loc_s1_ts_tensor.std(dim = 0)#.shape #.unsqueeze(0).repeat(5,1)
col_means[[0,1]] = 0
col_std[[0]] = 1
col_std[col_std==0] = 1
col_std[[1]] = 365 # normalize days by 365 -- each year ranges from 0 to 1
col_std[[-1]] = 90 # normalize angle by 90 -- angle ranges from 0 to 1

loc_s1_ts_tensor_std = col_std.unsqueeze(0).repeat(loc_s1_ts_tensor.shape[0],1)
loc_s1_ts_tensor_mean = col_means.unsqueeze(0).repeat(loc_s1_ts_tensor.shape[0],1)

loc_s1_ts_norm = (loc_s1_ts_tensor - loc_s1_ts_tensor_mean) / loc_s1_ts_tensor_std

# get max of number of observations per location
# idx = np.arange(loc_ts_norm.shape[0])
loc_id = np.unique(loc_s1_ts_norm[:,0])
num_obs = pd.DataFrame({'loc_id' : np.unique(loc_s1_ts_norm[:,0]).astype('int')})
num_obs['num_obs'] = [loc_s1_ts_norm[loc_s1_ts_norm[:,0]==i,:].shape[0] for i in num_obs['loc_id']]
print("Max number of observations for any loc_id")
print(num_obs.iloc[[num_obs['num_obs'].idxmax()]])

# %% Normalize s2 tensor
loc_s2_ts_tensor = loc_s2_ts_tensor[(loc_s2_ts_tensor[:,1] >= 0) & (loc_s2_ts_tensor[:,1] <= 365)]

row_means= loc_s2_ts_tensor.mean(dim = 1)#.shape #.unsqueeze(0).repeat(5,1)
loc_s2_ts_tensor = loc_s2_ts_tensor[~torch.isnan(row_means)]
col_means= loc_s2_ts_tensor.mean(dim = 0)#.shape #.unsqueeze(0).repeat(5,1)
col_std= loc_s2_ts_tensor.std(dim = 0)#.shape #.unsqueeze(0).repeat(5,1)
col_means[[0,1]] = 0
col_std[[0]] = 1
col_std[[1]] = 365 # normalize days by 365 -- each year ranges from 0 to 1

loc_s2_ts_tensor_std = col_std.unsqueeze(0).repeat(loc_s2_ts_tensor.shape[0],1)
loc_s2_ts_tensor_mean = col_means.unsqueeze(0).repeat(loc_s2_ts_tensor.shape[0],1)

loc_s2_ts_norm = (loc_s2_ts_tensor - loc_s2_ts_tensor_mean) / loc_s2_ts_tensor_std

# get max of number of observations per location
# idx = np.arange(loc_ts_norm.shape[0])
loc_id = np.unique(loc_s2_ts_norm[:,0])
num_obs = pd.DataFrame({'loc_id' : np.unique(loc_s2_ts_norm[:,0]).astype('int')})
num_obs['num_obs'] = [loc_s2_ts_norm[loc_s2_ts_norm[:,0]==i,:].shape[0] for i in num_obs['loc_id']]
print("Max number of observations for any loc_id")
print(num_obs.iloc[[num_obs['num_obs'].idxmax()]])


# In[86]:


loc_s2_ts_norm[1:5,:]




# In[61]:


loc_train


# ## Prepare the S2 dataset class


# In[91]:


# get y_train values from loc_train
y_train_df = (loc_train.assign(val = 1)   
              .pivot_table(columns = 'class', index = ['loc_id'], values = 'val', fill_value= 0)  
              .reset_index(['loc_id']))
y_train = y_train_df.to_numpy()
print('y_train:\n',y_train)

# get x_train values from loc_ts_norm (based on loc_id)
x_train = loc_ts_norm[torch.isin(loc_ts_norm[:,0],torch.tensor(y_train[:,0]).to(torch.float64)),:]

# get y_test values from loc_test
y_valid_df = (loc_valid.assign(val = 1)   
              .pivot_table(columns = 'class', index = ['loc_id'], values = 'val', fill_value= 0)   
              .reset_index(['loc_id']))
y_valid = y_valid_df.to_numpy()
print('y_valid:\n',y_valid[0:10,])

# get x_train values from loc_ts_norm (based on loc_id)
x_valid = loc_ts_norm[torch.isin(loc_ts_norm[:,0],torch.tensor(y_valid[:,0]).to(torch.float64)),:]

# get y_test values from loc_test
y_test_df = (loc_test.assign(val = 1)  
             .pivot_table(columns = 'class', index = ['loc_id'], values = 'val', fill_value= 0)  
             .reset_index(['loc_id']))
y_test = y_test_df.to_numpy()
print('y_test:\n',y_test[0:10,])

# get x_train values from loc_ts_norm (based on loc_id)
x_test = loc_ts_norm[torch.isin(loc_ts_norm[:,0],torch.tensor(y_test[:,0]).to(torch.float64)),:]

x_test


# ### build pytorch dataset: `s2_dateset`

# In[150]:


s2_train = s2Dataset(x = x_train, y = y_train, max_obs = 100)
s2_valid = s2Dataset(x = x_valid, y = y_valid, max_obs = 100)
s2_test = s2Dataset(x = x_test, y = y_test, max_obs = 100)

# example item in dataset
idx_test = 2
x, y = s2_train.__getitem__(idx_test)

print(f'x example, shape: {x.shape} \n(idx={idx_test}) columns: day, B8, B4, B3, B2\n',x)
# print()
print(f'\n\ny example (idx={idx_test}): crops(double) crops(single) plantation\n',y)
print(y.shape)
# sys.getsizeof(x)


# ### generate sampling weights for data loader

# In[151]:


# adapted from https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
target_classes = torch.stack([torch.argmax(s2_train.__getitem__(i)[1]) for i in range(s2_train.__len__())])
# count of samples in each class
class_sample_count = np.array([torch.sum(target_classes == i) for i in torch.unique(target_classes)])

# weight for each class (classed must go from 0 to n-1 classes)
weight = 1. / class_sample_count
sample_weights = np.array([weight[i] for i in target_classes])
sampler = WeightedRandomSampler(weights = sample_weights, num_samples = len(sample_weights))


# In[110]:


len(sample_weights)


# In[153]:


# s2_train

train_dl = DataLoader(s2_train, batch_size = 20, drop_last = True, sampler = sampler)
valid_dl = DataLoader(s2_valid, batch_size = 20, drop_last = False)
test_dl = DataLoader(s2_test, batch_size = 20, drop_last = False)


# In[113]:


len(train_dl)


# In[120]:


i = 1
for train, labels in train_dl:
    if i == 1:
        print("i == 1:\n",train[1, 1, :])
    if i == 10:
        print("i == 10:\n",train[1, 1, :])
    i += 1


# In[154]:


train_features, train_labels = next(iter(train_dl))
tf_test = train_features[:,:,:]
# tf_test
# train_labels
# tf_test
tf_test = tf_test.float()
print(tf_test.shape)

print(tf_test[0, 0:3, :])


# In[155]:


train_labels.shape


# In[20]:


# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)


# %%
assert 1 == 2

# %%
# In[156]:


import torch.nn as nn


nhead = 6 # number of attention heads
head_dim = 8 # dimension of each word for each attention head
dmodel = nhead * head_dim # embed_dim -- each word (row) is embedded to this dimension then split
# across the nhead attention heads

data_in = tf_test[:, :, 1:] # select only the data
positions = tf_test[:,:,0:1] # split out positional data
data_dim = data_in.shape[-1]


# In[22]:


torch.exp(torch.tensor([5.2333e-01]))/torch.sum(torch.exp(torch.tensor([-1.3249e-01, 5.2333e-01, -2.9124e-01])))


# In[157]:


from torch import nn, Tensor
class TransformerClassifier(nn.Module):
    def __init__(self, ntoken: int, dmodel: int, nhead: int, dhid: int, 
                 nlayers: int, data_dim: int, nclasses: int):
        """
        data_dim: dimension of data (i.e., num of columns) including position as first dimension
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
        classes = self.class_encoder(maxpool) #, dim = 1
        
        # classes = nn.functional.softmax(classes, 1) # don't use softmax with cross entropy loss... or do?
        # don't: https://stackoverflow.com/questions/55675345/should-i-use-softmax-as-output-when-using-cross-entropy-loss-in-pytorch
        # do: Machine Learning with Pytorch and Scikitlearn (p 471: Loss functions for classifiers) -- BUT NOT WITH CROSS ENTROPY LOSS (p478
        
        return classes

        # data_in = tf_test[:, :, 1:] # select only the data
        # positions = tf_test[:,:,0:1] # split out positional data
        # data_dim = data_in.shape[-1]
        
        
tfnetwork = TransformerClassifier(100, dmodel = 36, nhead = 6, dhid = 100, nlayers = 3, data_dim = 5, nclasses = 4)

tfnetwork(tf_test).shape


# In[158]:


from torchinfo import summary
print(tuple(tf_test.shape))
summary(tfnetwork, input_size = (5, 100, 5))


# In[234]:


train_features, train_labels = next(iter(train_dl))

tfnetwork = TransformerClassifier(100, dmodel = 36, nhead = 6, dhid = 100, nlayers = 3, data_dim = 5, nclasses = 4)

train_out = tfnetwork(train_features)


# In[163]:


train_labels.dtype


# In[229]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(tfnetwork.parameters(), lr = 0.001)

print(train_out.shape)

def get_num_correct(train_out, train_labels):
    pred = torch.argmax(train_out, dim = 1)
    actual = torch.argmax(train_labels, dim = 1)
    num_correct = torch.sum(pred == actual).item()
    # print('type',type(num_correct))
    # x = num_correct# item()
    # print('num_correct', num_correct.item())
    return num_correct


num_correct = get_num_correct(train_out, train_labels)
num_correct


# In[220]:


accuracy = num_correct / train_labels.size(0)
print('num_correct:', num_correct.item())
print('accuracy:', accuracy.item())
print('num in training sample:', train_labels.size(0))
tfnetwork.train()
loss = loss_fn(train_out, train_labels)
# loss.backward()
optimizer.step()
optimizer.zero_grad()
# tf_train
# tf_test.shape
f"accuracy: {accuracy.item()}"


# In[219]:


get_num_correct(train_out, train_labels).item()


# In[235]:


i = 1
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(tfnetwork.parameters(), lr = 0.001)


# print(i)
# for train_features, train_labels in train_dl:
#     i += 1
#     print(i)
n_epochs = 1000
loss_hist_train = [0] * n_epochs
accuracy_hist_train = [0] * n_epochs
loss_hist_valid = [0] * n_epochs
accuracy_hist_valid = [0] * n_epochs
for epoch in range(n_epochs):
    tfnetwork.train()
    for x_batch, y_batch in train_dl:
        
        # Forward pass
        pred = tfnetwork(x_batch)
        loss = loss_fn(pred, y_batch)
        
        loss_hist_train[epoch] += loss.item() * y_batch.size(0)
        accuracy_hist_train[epoch] += get_num_correct(pred, y_batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    loss_hist_train[epoch] /= float(len(train_dl.dataset))
    accuracy_hist_train[epoch] /= float(len(train_dl.dataset))
        

    
    # print('train_out.shape', train_out.shape)
    accuracy = get_accuracy(pred, y_batch)
    
    with torch.no_grad():
        for x_batch, y_batch in train_dl:

            # Forward pass
            pred = tfnetwork(x_batch)
            loss = loss_fn(pred, y_batch)

            loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
            accuracy_hist_valid[epoch] += get_num_correct(pred, y_batch)

        loss_hist_valid[epoch] /= float(len(train_dl.dataset))
        accuracy_hist_valid[epoch] /= float(len(train_dl.dataset))
        
    
    #     tfnetwork.eval()
    #     for x_batch, y_batch in valid_dl:
    #         # Forward pass
    #         pred = tfnetwork(x_batch)
    #         loss = loss_fn(pred, y_batch)

    #     tfnetwork.eval()
    # valid_features, valid_labels = /
    print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss_hist_train[epoch]:.4f}, Accuracy: {accuracy_hist_train[epoch]:.4f}'
          f' Val Accuracy: {accuracy_hist_valid[epoch]:.4f}')


# In[176]:


tfnetwork


# In[40]:


tfnetwork_out = tfnetwork(tf_test)
torch.max(tfnetwork_out,dim = 1)[0].shape


# In[25]:


torch.triu(torch.ones(4, 4) * float('-inf'), diagonal=0)


# In[41]:


dmodel


# ## Old S2 pytorch dataset

# In[ ]:


# class s2Dataset(Dataset):
#     """Sentinel 2 dataset"""
    
#     def __init__(self, proj_path, class_colname):
#         """
#         Args:
#             proj_path (string): path to manclassify project
#         """
#         self.proj_path = proj_path
#         proj_normpath = os.path.normpath(proj_path)
#         proj_dirname = proj_normpath.split(os.sep)[-1]
#         self.proj_name = re.sub("_classification$","",proj_dirname)
#         self.class_path = os.path.join(proj_path, self.proj_name + "_classification")
#         self.ts_path = os.path.join(proj_path, self.proj_name + "_download_timeseries")
#         self.pt_classes = pd.read_csv(os.path.join(self.class_path,"location_classification.csv"))
#         self.pt_classes = classes[['loc_id', class_colname]].dropna()
#         # self.pt_classes['loc_id'] = self.pt_classes['loc_id'] + 10.5 # for testing index only
#         self.classes = pd.unique(self.pt_classes[class_colname])
#         self.labels = self.pt_classes.assign(val = 1).pivot_table(columns = class_colname, index = 'loc_id', values = 'val', fill_value= 0)

    
#     def __getitem__(self, idx):
#         loc_id = self.labels.index[idx]
#         self.last_loc_id = loc_id
        
#         # select location id
#         s2_ts_x = s2_ts[['B8','B4','B3','B2','day']]
#         x = torch.tensor(s2_ts_x.to_numpy())
        
#         # get one-hot encoding for the point as tensor
#         y = torch.tensor(self.labels.iloc[idx].to_numpy())
        
#         return x, y
        
#     def __len__(self):
#         return self.pt_classes.shape[0]


# proj_path = "/Users/gopal/Google Drive/_Research/Research projects/ML/manclassify/app_data/Thailand"
# # date_rangeX = pd.to_datetime(['2019-06-01','2020-05-31'])
# s2_train = s2Dataset(proj_path = proj_path, class_colname = 'Subclass2019')
# x = s2_train.__getitem__(10)
# sys.getsizeof(x)

