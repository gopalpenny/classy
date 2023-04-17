#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 19:18:30 2023

@author: gopal

To run the script, at the command line type:
    
python train_s1_transformer.py path/to/s1_data.pt path/to/model_data_norms.pt path/to/model_data_labels.pt path/to/output_directory
python train_s1_transformer.py data/s1_data_prepped.pt data/model_data_norms.pt data/model_data_labels.pt s1_train_test
"""

# %%
import os
wds = ["/Users/gopalpenny/Projects/ml/classy/transformer_classifier",
       "/Users/gopal/Projects/ml/classy/transformer_classifier"]
wd_exists = [x for x in wds if os.path.exists(x)][0]
# os.chdir(wd_exists)

# %%
import torch
import sys
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml_transformer import TransformerClassifier, S1Dataset, scale_model_data


def train_transformer_s1(s1_data_path, norms_path, labels_path, output_dir_path):

    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    
    # %%
    s1_all = scale_model_data(data_path = "data/s1_data_prepped.pt", 
                              norms_path = "data/model_data_norms.pt",
                              data_name = "s1")
    labels = torch.load(labels_path)
    
    # %%
    # s1_all_old = torch.load(os.path.join(data_path, 'model_data_s1.pt'))
    # torch.equal(s1_all, s1_all_old)
    
    # %%
    y_train, y_eval = train_test_split(labels, train_size = 0.8, stratify = labels[:, 1])
    y_valid, y_test = train_test_split(y_eval, train_size = 0.5, stratify = y_eval[:, 1])
    
    print('y_train count [single, double, plantation, other]:')
    print([int(torch.sum(y_train[:,x + 1]).item()) for x in np.arange(labels.shape[1]-1)])
    
    print('y_valid count [single, double, plantation, other]:')
    print([int(torch.sum(y_valid[:,x + 1]).item()) for x in np.arange(labels.shape[1]-1)])
    
    print('y_test count [single, double, plantation, other]:')
    print([int(torch.sum(y_test[:,x + 1]).item()) for x in np.arange(labels.shape[1]-1)])
    
    # %%
    
    # Split training X values
    s1_train = s1_all[np.isin(s1_all[:, 0], y_train[:,0])]
    s1_valid = s1_all[np.isin(s1_all[:, 0], y_valid[:,0])]
    s1_test = s1_all[np.isin(s1_all[:, 0], y_test[:,0])]
    
    # %%
    
    # Prep datasets
    data_train = S1Dataset(s1_train, y_train, 64)
    data_valid = S1Dataset(s1_valid, y_valid, 64)
    data_test = S1Dataset(s1_test, y_test, 64)
    
    # %%
    
    # Prep weighted sampling for training data
    
    # adapted from https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
    target_labels = torch.stack([torch.argmax(data_train.__getitem__(i)[2]) for i in range(data_train.__len__())])
    # count of samples in each class
    class_sample_count = np.array([torch.sum(target_labels == i) for i in torch.unique(target_labels)])
    
    # weight for each class (labels must go from 0 to n-1 labels)
    weight = 1. / class_sample_count
    sample_weights = np.array([weight[i] for i in target_labels])
    sampler = WeightedRandomSampler(weights = sample_weights, num_samples = len(sample_weights))
    
    
    # %%
    # Prep dataloaders
    
    train_dl = DataLoader(data_train, batch_size = 20, drop_last = True, sampler = sampler)
    valid_dl = DataLoader(data_valid, batch_size = 20, drop_last = False)
    test_dl = DataLoader(data_test, batch_size = 20, drop_last = False)
    
    
    # %%
    # data_dim = 4: "day","VV","VH","angle" // loc_id included original data but not counted
    # because it doesn't get sent to the transformer
    
    xnn = TransformerClassifier(64, dmodel = 36, nhead = 6, dhid = 100, nlayers = 3, data_dim = 4, nclasses = 4)
    
    # %%
    # test transformer
    s1, y, _ = next(iter(train_dl))
    xnn(s1)
    
    # %%
    
    def get_num_correct(model_batch_out, y_batch):
        pred = torch.argmax(model_batch_out, dim = 1)
        actual = y_batch
        # actual = torch.argmax(y_batch, dim = 1)
        num_correct = torch.sum(pred == actual).item()
        # print('type',type(num_correct))
        # x = num_correct# item()
        # print('num_correct', num_correct.item())
        return num_correct
    
    
    # %%
    
    # Training loop
    
    # i = 1
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(xnn.parameters(), lr = 0.001)
    
    
    # print(i)
    # for train_features, train_labels in train_dl:
    #     i += 1
    #     print(i)
    n_epochs = 50
    loss_hist_train = [0] * n_epochs
    accuracy_hist_train = [0] * n_epochs
    loss_hist_valid = [0] * n_epochs
    accuracy_hist_valid = [0] * n_epochs
    for epoch in range(n_epochs):
        counter = 0
        xnn.train()
        for x_batch, y_batch, _ in train_dl:
            
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
            for x_batch, y_batch, _ in valid_dl:
    
                # Forward pass
                pred = xnn(x_batch)
                
                y_batch = y_batch.flatten().type(torch.LongTensor)
                
                # print('pred',pred)
                # print('target',y_batch)
                loss = loss_fn(pred, y_batch)
    
                # Accumulate loss and accuracy
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                accuracy_hist_valid[epoch] += get_num_correct(pred, y_batch)
    
            loss_hist_valid[epoch] /= float(len(valid_dl.dataset))
            accuracy_hist_valid[epoch] /= float(len(valid_dl.dataset))
            
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss_hist_train[epoch]:.4f}, Accuracy: {accuracy_hist_train[epoch]:.4f}'
              f' Val Accuracy: {accuracy_hist_valid[epoch]:.4f}')
        
    # %%
    loss_hist_test = 0
    accuracy_hist_test = 0
    with torch.no_grad():
        for x_batch, y_batch, _ in test_dl:
    
            # Forward pass
            pred = xnn(x_batch)
            
            y_batch = y_batch.flatten().type(torch.LongTensor)
            
            # print('pred',pred)
            # print('target',y_batch)
            loss = loss_fn(pred, y_batch)
    
            # Accumulate loss and accuracy
            loss_hist_test += loss.item() * y_batch.size(0)
            accuracy_hist_test += get_num_correct(pred, y_batch)
    
        loss_hist_test /= float(len(test_dl.dataset))
        accuracy_hist_test /= float(len(test_dl.dataset))
        
    print(f'Test Loss: {loss_hist_test:.4f}, Test Accuracy: {accuracy_hist_test:.4f}')
        
    
    torch.save(xnn, os.path.join(output_dir_path, "s1_xnn_trained.pt"))    
    
    # %%
    
    # test_dl
    # # %%
    # for i, x in enumerate(test_dl):   
    #     # print(i)
    #     print(len(x['x']))
        # print(y_batch.shape)
    
    # %%
    fig, axs = plt.subplots(2)
    axs[0].plot(loss_hist_train, label = "Training")
    axs[0].plot(loss_hist_valid, label = "Validation")
    axs[0].set(ylabel = "Loss")
    axs[0].legend()
    axs[1].plot(accuracy_hist_train)
    axs[1].plot(accuracy_hist_valid)
    axs[1].set(ylabel = "Accuracy", xlabel = "Epoch")
    
    # plt.show()
    
    # %%
    plt.savefig(os.path.join(output_dir_path, "s1_training_loss.png"))
    
    # %%
    training_metrics = pd.DataFrame({
        'loss_hist_train' : loss_hist_train,
        'loss_hist_valid' : loss_hist_valid,
        'accuracy_hist_train' : accuracy_hist_train,
        'accuracy_hist_valid' : accuracy_hist_valid,})
    
    training_metrics.to_csv(os.path.join(output_dir_path,"s1_training_metrics.csv"))
    
if __name__ == "__main__":
    
    cmd_args = sys.argv
    print(f'Number of input arguments: {len(cmd_args)-1}.\n')
    
    num_user_args = 5

    if len(cmd_args) == num_user_args:
        s1_data_path = cmd_args[1]
        norms_path = cmd_args[2]
        labels_path = cmd_args[3]
        output_dir_path = cmd_args[4]
        print('User-defined input arguments:\n')
    else:
    
        s1_data_path = "data/s1_data_prepped.pt"
        labels_path = 'data/model_data_labels.pt'
        norms_path = "data/model_data_norms.pt"
        output_dir_path = "./s1_train"
        
        print(f'For defining custom arguments, specify {num_user_args-1} inputs.')
        print('Using default input arguments:\n')
        
    print(f's1_data_path: {s1_data_path}')
    print(f'labels_path: {labels_path}')
    print(f'norms_path: {norms_path}')
    print(f'output_dir_path: {output_dir_path}\n')
    print('Running predict_s1_classes() function...')
    # time.sleep(1)
    
    train_transformer_s1(s1_data_path, norms_path, labels_path, output_dir_path)
    

