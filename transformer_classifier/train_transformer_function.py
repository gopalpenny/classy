#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 19:18:30 2023

@author: gopal

To run the script, at the command line type:
    
python train_s1_transformer.py path/to/s1_data.pt path/to/model_data_norms.pt path/to/model_data_labels.pt path/to/output_directory 25
python train_s1_transformer.py data/s1_data_prepped.pt data/model_data_norms.pt data/model_data_labels.pt s1_train_test 25
"""
# %%
# torch.ones(3,4)
# torch.zeros(0,4)

# torch.cat((torch.ones(3,4), torch.zeros(0,4)), dim = 0)

# %%
import os
wds = ["/Users/gopalpenny/Projects/ml/classy/transformer_classifier",
       "/Users/gopal/Projects/ml/classy/transformer_classifier"]
# wd_exists = [x for x in wds if os.path.exists(x)][0]
# os.chdir(wd_exists)

# %%
import torch
import sys
import time
from torch import nn
import re
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import importlib
# importlib.reload(sys.modules['ml_transformer'])
from transformer_sentinel import TransformerClassifier, SentinelDataset, scale_model_data

# %%
if False:
    # %%
    s1_data_path = "data/s1_data_prepped.pt"
    s2_data_path = None # "data/s2_data_prepped.pt"
    labels_path = 'data/model_data_labels.pt'
    norms_path = "data/model_data_norms.pt"
    output_dir_path = "./s1_train"
# %%
def train_transformer_func(xnn, s1_data_path, s2_data_path, norms_path, labels_path, output_dir_path, n_epochs, batch_size, lr, weight_decay):

    # %%
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)



    print_path = os.path.join(output_dir_path, "output.txt")
    flog = Logger(print_path)

    flog.write(locals(), 'Sending output to {output_dir_path}')
    flog.write(locals(), 'Writing logfile to output.txt')

    flog.write(locals(), 'Beginning train_transformer_func() function with {n_epochs} epochs...')
    # orig_stdout = sys.stdout
    # print_out = open(print_path, 'w')
    # sys.stdout = print_out

    # # %%
    # if torch.backends.mps.is_available():
    #     # device = torch.device("mps")
    #     device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    xnn.to(device)
    flog.write(locals(), 'Beginning train_transformer_func() function...\n')

    t_start = time.time()

    # with open(print_path, 'w') as f:
    flog.write(locals(), "starting training at {time.strftime('%c')} on {device}", {'time': time})    
    
    # %%

    labels = torch.load(labels_path)

    y_train, y_eval = train_test_split(labels, train_size = 0.8, stratify = labels[:, 1])
    y_valid, y_test = train_test_split(y_eval, train_size = 0.5, stratify = y_eval[:, 1])

    flog.write(locals(), 'Number of training samples: {y_train.shape[0]}')
    flog.write(locals(), 'Number of validation samples: {y_valid.shape[0]}')
    flog.write(locals(), 'Number of test samples: {y_test.shape[0]}')

    # %%

    if s1_data_path is not None:
        print("Reading in Sentinel 1 data and scaling based on norms...")
        s1_all = scale_model_data(data_path = s1_data_path, 
                                    norms_path = norms_path,
                                    data_name = "s1")
        s1_all = s1_all
        
        # Split training X values
        s1_train = s1_all[np.isin(s1_all[:, 0], y_train[:,0])]
        s1_valid = s1_all[np.isin(s1_all[:, 0], y_valid[:,0])]
        s1_test = s1_all[np.isin(s1_all[:, 0], y_test[:,0])]
    else:
        print("No Sentinel 1 data used.")
        s1_train = None
        s1_valid = None
        s1_test = None

    
    if s2_data_path is not None:
        print("Reading in Sentinel 2 data and scaling based on norms...")
        s2_all = scale_model_data(data_path = s2_data_path,
                                    norms_path = norms_path,
                                    data_name = "s2")
        s2_all = s2_all

        s2_train = s2_all[np.isin(s2_all[:, 0], y_train[:,0])]
        s2_valid = s2_all[np.isin(s2_all[:, 0], y_valid[:,0])]
        s2_test = s2_all[np.isin(s2_all[:, 0], y_test[:,0])]
    else:
        print("No Sentinel 2 data used.")
        s2_train = None
        s2_valid = None
        s2_test = None

    # %%

    # Prep datasets
    data_train = SentinelDataset(y_train, s1_train, s2_train)
    data_valid = SentinelDataset(y_valid, s1_valid, s2_valid)
    data_test = SentinelDataset(y_test, s1_test, s2_test)

    # %%
    # check training dataset returns correct values
    # a = data_train.__getitem__(0)
    # [str(a[i].shape) for i in range(len(a)) if a[i] is not None]

    # %%

    # Prep weighted sampling for training data

    # adapted from https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
    target_labels = torch.stack([data_train.__getitem__(i)[2] for i in range(data_train.__len__())])
    valid_labels = torch.stack([data_valid.__getitem__(i)[2] for i in range(data_valid.__len__())])
    test_labels = torch.stack([data_test.__getitem__(i)[2] for i in range(data_test.__len__())])
    # count of samples in each training class
    class_sample_count = np.array([torch.sum(target_labels == i) for i in torch.unique(target_labels)])

    # %%
    flog.write(locals(), 'Number of samples in each class:\n' +
        '[0: single, 1: double, 2: plantation, 3: other]')
    flog.write(locals(), 'Training: {class_sample_count}')
    valid_counts = np.array([torch.sum(valid_labels == i) for i in torch.unique(valid_labels)])
    test_counts = np.array([torch.sum(test_labels == i) for i in torch.unique(test_labels)])
    flog.write(locals(), 'Valid: {valid_counts}')
    flog.write(locals(), 'Test: {test_counts}')

    # %%
    # weight for each class (labels must go from 0 to n-1 labels)
    weight = 1. / class_sample_count
    sample_weights = np.array([weight[i] for i in target_labels.int()])
    sampler = WeightedRandomSampler(weights = sample_weights, num_samples = len(sample_weights))

    # %%
    # Custom collate function ## NOT WORKING
    def collate_batches(batch):
        s1_seq = [item[0] for item in batch]
        s1_padded = pad_sequence(s1_seq, batch_first=True)
        s2_seq = [item[1] for item in batch]
        s2_padded = pad_sequence(s2_seq, batch_first=True)
        y_seq = [item[2] for item in batch]
        y_padded = pad_sequence(y_seq, batch_first=True)
        locid_seq = [item[3] for item in batch]
        locid_padded = torch.Tensor(locid_seq).float() #pad_sequence(locid_seq, batch_first=True)
        # sequences = [torch.tensor(item) for item in batch]
        return s1_padded, s2_padded, y_padded, locid_padded

    # Prep dataloaders
    train_dl = DataLoader(data_train, batch_size = batch_size, drop_last = True, sampler = sampler, collate_fn = collate_batches)
    valid_dl = DataLoader(data_valid, batch_size = batch_size, drop_last = False, collate_fn = collate_batches)
    test_dl = DataLoader(data_test, batch_size = batch_size, drop_last = False, collate_fn = collate_batches)

    # %%
    # test transformer
    # a = next(iter(train_dl))
    # s1, s2, y, _ = next(iter(train_dl))
    # flog.write(locals(), '{s1.shape}')
    # flog.write(locals(), '{s2_2.shape}')
    # flog.write(locals(), '{y.shape}')

    # %%
    # s2_2.shape
    # collate_batches([s2, s2_2])
    # xnn(s2)


    # %%
    # data_dim = 4: "day","VV","VH","angle" // loc_id included original data but not counted
    # because it doesn't get sent to the transformer

    # xnn = TransformerClassifier(dmodel = 36, nhead = 6, dhid = 100, nlayers = 3, s1_dim = 4, s2_dim = 5, nclasses = 4)

    # %%
    # test transformer
    s1, s2, y, _ = next(iter(train_dl))
    # s2x = s2[:, :, 0:4]
    # xnn(s1, s2)

    # %%

    def get_num_correct(model_batch_out, y_batch):
        pred = torch.argmax(model_batch_out, dim = 1)
        actual = y_batch
        # actual = torch.argmax(y_batch, dim = 1)
        num_correct = torch.sum(pred == actual).item()
        # flog.write(locals(), 'type',type(num_correct))
        # x = num_correct# item()
        # flog.write(locals(), 'num_correct', num_correct.item())
        return num_correct


    # %%

    # Training loop

    # i = 1
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(xnn.parameters(), lr = lr, weight_decay = weight_decay)


    # flog.write(locals(), i)
    # for train_features, train_labels in train_dl:
    #     i += 1
    #     flog.write(locals(), i)
    # n_epochs = 50
    loss_hist_train = [0] * n_epochs
    accuracy_hist_train = [0] * n_epochs
    loss_hist_valid = [0] * n_epochs
    accuracy_hist_valid = [0] * n_epochs

    # %%
    for epoch in range(n_epochs):
        
        counter = 0
        xnn.train()

        for s1_batch, s2_batch, y_batch, loc_id in train_dl:
            
            # loading to device for each batch is not uncommon, see
            # https://stackoverflow.com/questions/60789801/moving-dataloader-data-to-gpu-pytorch
            # s1_batch = s1_batch.to(device)
            # s2_batch = s2_batch.to(device)
            # y_batch = y_batch.to(device)
            
            # Forward pass
            pred = xnn(s1_batch, s2_batch)
            
            y_batch = y_batch.flatten().type(torch.LongTensor)
            # flog.write(locals(), y_batch)

            # print first tensor for debugging
            # if epoch == 0 and counter == 0:
            #     flog.write(locals(), pred)
            #     flog.write(locals(), y_batch)
            #     counter +=1
                
            loss = loss_fn(pred, y_batch)
            
            # Accumulate loss and accuracy
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            # flog.write(locals(), 'counter: {counter} loc_id: {loc_id}: {loss.item()} pred: {pred} y_batch: {y_batch}')
            accuracy_hist_train[epoch] += get_num_correct(pred, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counter +=1   
        
        loss_hist_train[epoch] /= float(len(train_dl.dataset))
        accuracy_hist_train[epoch] /= float(len(train_dl.dataset))
        
        with torch.no_grad():
            for s1_batch, s2_batch, y_batch, _ in valid_dl:

                # Forward pass
                pred = xnn(s1_batch, s2_batch)
                
                y_batch = y_batch.flatten().type(torch.LongTensor)
                
                # flog.write(locals(), 'pred',pred)
                # flog.write(locals(), 'target',y_batch)
                loss = loss_fn(pred, y_batch)

                # Accumulate loss and accuracy
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                accuracy_hist_valid[epoch] += get_num_correct(pred, y_batch)

            loss_hist_valid[epoch] /= float(len(valid_dl.dataset))
            accuracy_hist_valid[epoch] /= float(len(valid_dl.dataset))
            
        flog.write(locals(), 'Epoch [{epoch+1}/{n_epochs}], Loss: {loss_hist_train[epoch]:.4f}, ' + 
                   'Accuracy: {accuracy_hist_train[epoch]:.4f} ' +
                   'Val Accuracy: {accuracy_hist_valid[epoch]:.4f}')
        
    # %%

    loss_hist_test = 0
    accuracy_hist_test = 0
    with torch.no_grad():
        for s1_batch, s2_batch, y_batch, _ in test_dl:

            # Forward pass
            pred = xnn(s1_batch, s2_batch)
            
            y_batch = y_batch.flatten().type(torch.LongTensor)
            
            # flog.write(locals(), 'pred',pred)
            # flog.write(locals(), 'target',y_batch)
            loss = loss_fn(pred, y_batch)

            # Accumulate loss and accuracy
            loss_hist_test += loss.item() * y_batch.size(0)
            accuracy_hist_test += get_num_correct(pred, y_batch)

        loss_hist_test /= float(len(test_dl.dataset))
        accuracy_hist_test /= float(len(test_dl.dataset))
        
    flog.write(locals(), 'Test Loss: {loss_hist_test:.4f}, Test Accuracy: {accuracy_hist_test:.4f}')
        

    torch.save(xnn, os.path.join(output_dir_path, "xnn_trained.pt"))    

   # %%
    t_end = time.time()
    t_diff_sec = t_end - t_start
    flog.write(locals(), "finished training at {time.strftime('%c')}", {'time': time})
    flog.write(locals(), "total time: {time.strftime('%HH:%MM:%SS', time.gmtime(t_diff_sec))}", {'time': time})


    # %%

    # test_dl
    # # %%
    # for i, x in enumerate(test_dl):   
    #     # flog.write(locals(), i)
    #     flog.write(locals(), len(x['x']))
        # flog.write(locals(), y_batch.shape)
    
    flog.write(locals(), "Saving training figure and data...")
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
    plt.savefig(os.path.join(output_dir_path, "training_loss.png"))

    # %%
    training_metrics = pd.DataFrame({
        'loss_hist_train' : loss_hist_train,
        'loss_hist_valid' : loss_hist_valid,
        'accuracy_hist_train' : accuracy_hist_train,
        'accuracy_hist_valid' : accuracy_hist_valid,})

    training_metrics.to_csv(os.path.join(output_dir_path,"training_metrics.csv"))
    flog.write(locals(), "Done.\n")

    # sys.stdout = orig_stdout
    # print_out.close()

def get_fstring(fstring, locals, globals = None):

    # print(fstring)
    fstring_clean = re.sub('\n', 'ENDOFLINE', fstring)
    ret_val_clean = eval(f'f"{fstring_clean}"', locals, globals)
    ret_val = re.sub('ENDOFLINE', '\n', ret_val_clean)
    return ret_val

class Logger(object):
    def __init__(self, out_path):
        self.terminal = sys.stdout
        self.log = open(out_path, "w")
   
    def write(self, locals, fstring, globals = None):
        message = get_fstring(fstring, locals, globals)
        self.terminal.write(message + '\n')
        self.log.write(message + '\n')

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass 
    
if __name__ == "__main__":
    
    cmd_args = sys.argv
    print(f'Number of input arguments: {len(cmd_args)-1}.\n')
    
    num_user_args = 5

    if len(cmd_args) == num_user_args:
        s1_data_path = cmd_args[1]
        norms_path = cmd_args[2]
        labels_path = cmd_args[3]
        output_dir_path = cmd_args[4]
        n_epochs = int(cmd_args[5])
        print('User-defined input arguments:\n')
    else:
    
        s1_data_path = "data/s1_data_prepped.pt"
        s2_data_path = "data/s2_data_prepped.pt"
        labels_path = 'data/model_data_labels.pt'
        norms_path = "data/model_data_norms.pt"
        output_dir_path = "./s1_train"
        n_epochs = 25
        
        print('For defining custom arguments, specify {num_user_args-1} inputs.')
        print('Using default input arguments:\n')
        
    print('s1_data_path: {s1_data_path}')
    print('s2_data_path: {s2_data_path}')
    print('labels_path: {labels_path}')
    print('norms_path: {norms_path}')
    print('output_dir_path: {output_dir_path}\n')
    print('n_epochs: {n_epochs}\n')
    print('Running predict_s1_classes() function...')
    # time.sleep(1)
    
    xnn = TransformerClassifier(dmodel = 36, nhead = 6, dhid = 100, nlayers = 3, s1_dim = 4, s2_dim = 5, nclasses = 4)

    train_transformer_func(xnn, s1_data_path, s2_data_path, norms_path, labels_path, output_dir_path, n_epochs)
    


# %%
