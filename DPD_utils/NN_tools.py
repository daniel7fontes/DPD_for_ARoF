# -*- coding: utf-8 -*-
"""
======================================================================
Funções para aplicação, treinamento e verificação de desempenho da NN-DPD
======================================================================
"""

import numpy as np
import torch as th

from tqdm.notebook  import tqdm
from scipy.constants import pi
from torch           import nn

from torch.utils.data         import Dataset, DataLoader
from optic_private.torchDSP   import pnorm
from optic_private.torchUtils import memoryLessDataSet, MLP, train_model, test_model, slidingWindowDataSet, fitFilterNN


def NN_training(sigIn, sigRef, param, RoFChannel_model = None):
    
    num_feat = param.num_input
    
    N1 = param.N1
    N2 = param.N2
    
    divByL        = param.divByL
    trainTestFrac = param.trainTestFrac
    batch_size    = param.batch_size
    shuffle       = param.shuffle
    includeMemory = param.includeMemory
    Ntaps         = param.Ntaps
    augment       = param.augment
    
    lr            = param.lr
    epochs        = param.epochs
    
    pgrsBar       = param.pgrsBar
    directLearn   = param.directLearn
    device        = param.device

    if directLearn:
        train_dataloader, test_dataloader = createDatasets(sigRef, sigRef, divByL, trainTestFrac,\
                                                           batch_size, includeMemory, Ntaps, augment=augment)

        for p in RoFChannel_model.parameters():
            p.requires_grad = False
        
        if includeMemory:
            DPD_model = MLP([num_feat*Ntaps, N1, N2, 2], activation=nn.ReLU()).to(device)
        else:
            DPD_model = MLP([2, 32, 32, 2], activation=nn.ReLU()).to(device)
        
        
        loss_fn = nn.MSELoss()
        optimizer = th.optim.Adam(DPD_model.parameters(), lr = lr)
        
        trainLoss = np.zeros(epochs)
        testLoss  = np.zeros(epochs)
        
        for t in tqdm(range(epochs), disable = not(pgrsBar)):
            # Training
            size = len(train_dataloader.dataset)
            DPD_model.train()
            
            for batch, (X, y) in enumerate(train_dataloader):    
                # Compute prediction error        
                chInput = DPD_model(X)                
                if includeMemory:
                    chInput = th.view_as_complex(chInput)            
                    chOutput = th.view_as_real(fitFilterNN(chInput, RoFChannel_model, Ntaps, 1, len(chInput), predict=False))
                    loss = loss_fn(chOutput, y)
                else:
                    chOutput = RoFChannel_model(chInput)            
                    loss = loss_fn(chOutput, X)
        
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                if batch % 100 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    #logg.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            # Validation      
            num_batches = len(test_dataloader)
            DPD_model.eval()
            test_loss = 0    
            with th.no_grad():
                for X, y in test_dataloader:           
                    # Compute prediction error        
                    chInput = DPD_model(X)           
                    
                    if includeMemory:
                        chInput = th.view_as_complex(chInput)
                        chOutput = th.view_as_real(fitFilterNN(chInput, RoFChannel_model, Ntaps, 1, len(chInput)))
                        test_loss += loss_fn(chOutput, y).item()
                    else:
                        chOutput = RoFChannel_model(chInput)           
                        test_loss += loss_fn(chOutput, X).item()          
                                    
            test_loss /= num_batches    
            
            trainLoss[t] = loss.item()
            testLoss[t]  = test_loss
            
            if (t+1) % 100 == 0:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']/1.25

    
    else:
        train_dataloader, test_dataloader = createDatasets(sigRef, sigIn, divByL, trainTestFrac,\
                                                   batch_size, includeMemory, Ntaps, augment=augment)
        
        # Define neural network (multilayer perceptron - MLP) model
        if includeMemory:
            DPD_model = MLP([num_feat*Ntaps, N1, N2, 2], activation = nn.ReLU()).to(device)
        else:
            DPD_model = MLP([2, 32, 32, 2], activation = nn.ReLU()).to(device)
        
        loss_fn = nn.MSELoss()
        optimizer = th.optim.Adam(DPD_model.parameters(), lr = lr)
        
        trainLoss = np.zeros(epochs)
        testLoss  = np.zeros(epochs)
        
        for t in tqdm(range(epochs), disable = not(pgrsBar)): 
            train_losses = train_model(train_dataloader, DPD_model, loss_fn, optimizer)
            test_losses  = test_model(test_dataloader, DPD_model, loss_fn)    
            
            trainLoss[t] = np.mean(train_losses)
            testLoss[t]  = np.mean(test_losses)
            
            if (t+1)%100 == 0:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']/2

        
    return DPD_model, trainLoss, testLoss



def createDatasets(
    sigIn,
    sigRef,
    divByL,
    trainTestFrac,
    batch_size,
    includeMemory,
    Ntaps,
    shuffle=False,
    augment=False
):
    sig_in  = pnorm(sigIn[0 : len(sigIn) // divByL])
    sig_ref = pnorm(sigRef[0 : len(sigRef) // divByL])

    # Create the dataset
    indx_train = th.arange(0, int(trainTestFrac * len(sig_in)))
    indx_test = th.arange(int(trainTestFrac * len(sig_in)), len(sig_in))

    sig_train = sig_in[indx_train]  # get signal amplitude samples (L,)
    sig_test = sig_in[indx_test]  # get signal amplitude samples (L,)

    indy_train = th.arange(0, int(trainTestFrac * len(sig_ref)))
    indy_test = th.arange(int(trainTestFrac * len(sig_ref)), len(sig_ref))
       
    if includeMemory:
        train_dataset = slidingWindowDataSet(
            sig_ref[indy_train], sig_train, Ntaps, augment=augment
        )
        test_dataset = slidingWindowDataSet(
            sig_ref[indy_test], sig_test, Ntaps, augment=augment
        )

        # Create a data loader for batching and shuffling the data
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=shuffle
        )
    else:
        train_dataset = memoryLessDataSet(sig_ref[indy_train], sig_train, augment=augment)
        test_dataset = memoryLessDataSet(sig_ref[indy_test], sig_test, augment=augment)

        # Create a data loader for batching and shuffling the data
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=shuffle
        )

    return train_dataloader, test_dataloader
