#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:09:43 2023

@author: gopal
"""

# ml_predict.py

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml_transformer import SentinelDatasets, TransformerClassifier

# %%
# os.chdir("/Users/gopalpenny/Projects/ml/classy/classapp/transformer_classifier")
os.chdir("/Users/gopal/Projects/ml/classy/classapp/transformer_classifier")

# %%

data_path = "./data"
sim_path = "./sim"

if not os.path.exists(sim_path):
    os.mkdir(sim_path)


s1_all = torch.load(os.path.join(data_path, 'model_data_s1.pt'))
s2_all = torch.load(os.path.join(data_path, 'model_data_s2.pt'))
labels = torch.load(os.path.join(data_path, 'model_data_labels.pt'))

# %%


# %%

# Prep datasets
data_predict = SentinelDatasets(s1_all, s2_all, labels, 64, 64)

# %%
# Prep dataloaders

dl = DataLoader(data_predict, batch_size = 20, drop_last = False)


# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

xnn = torch.load(os.path.join(sim_path, 'xnn_trained.pt'))
xnn = xnn.to(device)
xnn.eval();

# %%
# test transformer
# s1, s2, y = next(iter(dl))
# xnn(s2)
# %%

# Prediction loop

counter = 0
xnn.eval()

for _, x_batch, y_batch in dl:
    
    # Forward pass
    pred = xnn(x_batch)
    
    y_batch = y_batch.flatten().type(torch.LongTensor)
    
    # print first tensor for debugging
    # if epoch == 0 and counter == 0:
    #     print(pred)
    #     print(y_batch)
    #     counter +=1
        
    loss = loss_fn(pred, y_batch)
    
    
    # Accumulate loss and accuracy
    loss_hist_train[epoch] += loss.item() * y_batch.size(0)
    
    accuracy_hist_train[epoch] += get_num_correct(pred, y_batch)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
loss_hist_train[epoch] /= float(len(train_dl.dataset))
accuracy_hist_train[epoch] /= float(len(train_dl.dataset))

with torch.no_grad():
    for _, x_batch, y_batch in train_dl:

        # Forward pass
        pred = xnn(x_batch)
        
        y_batch = y_batch.flatten().type(torch.LongTensor)
        
        # print('pred',pred)
        # print('target',y_batch)
        loss = loss_fn(pred, y_batch)

        # Accumulate loss and accuracy
        loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
        accuracy_hist_valid[epoch] += get_num_correct(pred, y_batch)

    loss_hist_valid[epoch] /= float(len(train_dl.dataset))
    accuracy_hist_valid[epoch] /= float(len(train_dl.dataset))
    
print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss_hist_train[epoch]:.4f}, Accuracy: {accuracy_hist_train[epoch]:.4f}'
      f' Val Accuracy: {accuracy_hist_valid[epoch]:.4f}')