#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

from Pyfhel import Pyfhel, PyPtxt, PyCtxt

import torch
import torch.nn as nn

import time
import os
import sys

device = "cpu"
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path) 

from pycrcnn.net_builder.encoded_net_builder_ts import build_from_pytorch
from pycrcnn.crypto.crypto import encrypt_matrix, decrypt_matrix
from train_utils import *

N_EXPERIMENTS = 10

import logging as log
log.basicConfig(filename='times_pinpoint_2conv.log',
                format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S', level=log.DEBUG)


# # Models

# In[2]:


class Square(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, t):
        return torch.pow(t, 2)
    
class PINPOINT_1CONV(nn.Module):
    """Convolutional Neural Networks"""
    def __init__(self, input_size, output_horizon):
        super(PINPOINT_1CONV, self).__init__()

        n_kernels_1 = 32
        kernel_size_1 = 3
        out_conv_1 = n_kernels_1 * (input_size - kernel_size_1 + 1)

        self.main = nn.Sequential(           
            nn.Conv1d(in_channels=1, out_channels=n_kernels_1, kernel_size=kernel_size_1),
            Square(),
            nn.Flatten(),      
            
            nn.Linear(out_conv_1, int(out_conv_1/2)), #use without avgpool
            nn.Linear(int(out_conv_1/2), int(out_conv_1/4)),
            nn.Linear(int(out_conv_1/4), output_horizon)   
        )

    def forward(self, x):
        out = self.main(x)
        return out
    
    def __str__(self):
        return "PINPOINT_Small"

    
class PINPOINT_2CONV(nn.Module):
    """Convolutional Neural Networks"""
    def __init__(self, input_size, output_horizon):
        super(PINPOINT_2CONV, self).__init__()
        
        n_kernels_1 = 16
        n_kernels_2 = 32
        kernel_size_1 = 5
        kernel_size_2 = 3
        
        out_conv_1 = input_size - kernel_size_1 + 1
        out_conv_2 = n_kernels_2 * (out_conv_1 - kernel_size_2 + 1)

        self.main = nn.Sequential(           
            nn.Conv1d(in_channels=1, out_channels=n_kernels_1, kernel_size=kernel_size_1),
            Square(),
            nn.Conv1d(in_channels=n_kernels_1, out_channels=n_kernels_2, kernel_size=kernel_size_2),
            Square(),
            nn.Flatten(),      
            
            nn.Linear(out_conv_2, int(out_conv_2/2)), #use without avgpool
            nn.Linear(int(out_conv_2/2), int(out_conv_2/4)),
            nn.Linear(int(out_conv_2/4), output_horizon)   
        )

    def forward(self, x):
        out = self.main(x)
        return out
    
    def __str__(self):
        return "PINPOINT_Medium"


model_input = np.array([[0.5] for _ in range(0, 14)]).reshape(1, 1, 14)


def encrypted_processing(model, m, p):   
    
    times_plain = np.array([])
    for _ in range(0, N_EXPERIMENTS):
        t0 = time.time()
        with torch.set_grad_enabled(False):
            expected_output = model(torch.FloatTensor(model_input))
            
        times_plain = np.append(times_plain, time.time()-t0)
        
    log.info(f"Using encryption parameters: m={m}, p={p}")
    
    times_encrypted = np.array([])
    
    for i in range(0, N_EXPERIMENTS):
    
        HE = Pyfhel()    
        HE.contextGen(p=p, m=m, intDigits=16, fracDigits=128) 
        HE.keyGen()
        HE.relinKeyGen(30, 3)

        encoded_model = build_from_pytorch(HE, model.main)
        encrypted_input = encrypt_matrix(HE, model_input)

        t0 = time.time()

        for layer in encoded_model:
            encrypted_input = layer(encrypted_input)

        times_encrypted = np.append(times_encrypted, time.time()-t0)

        result = decrypt_matrix(HE, encrypted_input)

        assert np.allclose(expected_output.numpy(), result,
                           rtol=1e-01, atol=1e-01)
    
    log.info(f"Mean time requested over {N_EXPERIMENTS} (plain processing): {np.mean(times_plain):.2f}")
    log.info(f"Mean time requested over {N_EXPERIMENTS} (encrypted processing): {np.mean(times_encrypted):.2f}")


log.info("Trying with PINPOINT-1CONV...")

model = PINPOINT_2CONV(14, 7)
p=615535178676
m=8192

encrypted_processing(model, m, p)


