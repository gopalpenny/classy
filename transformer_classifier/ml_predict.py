#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:09:43 2023

@author: gopal
"""

# ml_predict.py

# %%
import os
# os.chdir("/Users/gopalpenny/Projects/ml/classy/classapp/transformer_classifier")
# os.chdir("/Users/gopal/Projects/ml/classy/classapp/transformer_classifier")

# %%
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml_transformer import SentinelDatasets, TransformerClassifier


# %%

data_path = "./data"
sim_path = "./sim"
predict_path = "./predict"

if not os.path.exists(sim_path):
    os.mkdir(sim_path)
if not os.path.exists(predict_path):
    os.mkdir(predict_path)


# Load data
s1_all = torch.load(os.path.join(data_path, 'model_data_s1.pt'))
s2_all = torch.load(os.path.join(data_path, 'model_data_s2.pt'))

# create empty labels tensor with pt_ids (for Dataset & DataLoader)
# This answer suggests using Dataloader even for eval mode: https://stackoverflow.com/a/73396570
pt_ids_all = torch.unique(s1_all[:, 0])
labels = torch.zeros((pt_ids_all.shape[0],2))
labels[:, 0] = pt_ids_all

# %%

# Prep datasets
data_predict = SentinelDatasets(s1_all, s2_all, labels, 64, 64)

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

pred_labels = torch.zeros((0, 2))
counter = 0
xnn.eval()

for _, x_batch, y_batch, loc_id in dl:
    
    model_batch_out = xnn(x_batch)
    
    pred = torch.argmax(model_batch_out, dim = 1)
    
    torch.stack((loc_id, pred))
    pred_labels = torch.concat((pred_labels, torch.stack((loc_id, pred), dim = 1)))

# %%
torch.save(pred_labels, os.path.join(predict_path, 'predictions.pt'))