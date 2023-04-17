#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:09:43 2023

@author: gopal
"""

# ml_predict.py

# %%
import os
wds = ["/Users/gopalpenny/Projects/ml/classy/transformer_classifier",
       "/Users/gopal/Projects/ml/classy/transformer_classifier"]
wd_exists = [x for x in wds if os.path.exists(x)][0]
# os.chdir(wd_exists)

# %%
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml_transformer import TransformerClassifier, S1Dataset


# %%

def predict_s1_classes(s1_data_path, trained_model_path, output_dir_path):
    
    if not os.path.exists(predict_path):
        os.mkdir(predict_path)
    
    # Load data
    s1_all = torch.load(s1_data_path)
    
    # create empty labels tensor with pt_ids (for Dataset & DataLoader)
    # This answer suggests using Dataloader even for eval mode: https://stackoverflow.com/a/73396570
    pt_ids_all = torch.unique(s1_all[:, 0])
    labels = torch.zeros((pt_ids_all.shape[0],2))
    labels[:, 0] = pt_ids_all
    
    # %%
    
    # Prep datasets
    data_predict = S1Dataset(s1_all, labels, 64)
    
    # Prep dataloaders
    dl = DataLoader(data_predict, batch_size = 20, drop_last = False)
    
    # %%
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # %%
    
    xnn = torch.load(trained_model_path)
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
    
    for x_batch, y_batch, loc_id in dl:
        
        model_batch_out = xnn(x_batch)
        
        pred = torch.argmax(model_batch_out, dim = 1)
        
        torch.stack((loc_id, pred))
        pred_labels = torch.concat((pred_labels, torch.stack((loc_id, pred), dim = 1)))
    
    # %%
    torch.save(pred_labels, os.path.join(output_dir_path, 'predictions.pt'))
    

if __name__ == '__main__':
    
    data_path = "./data"
    model_path = "./s1_train"
    trained_model_path = os.path.join(model_path, 's1_xnn_trained.pt')
    output_dir_path = "./predict"
    s1_data_path = os.path.join(data_path, 'model_data_s1.pt')
    
    predict_s1_classes(s1_data_path, trained_model_path, output_dir_path)
    
    
    