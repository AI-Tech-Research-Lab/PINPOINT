import torch.nn as nn
import torch
from math import floor

class Square(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, t):
        return torch.pow(t, 2)

class Cube(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, t):
        return torch.pow(t, 3)
    
class Printer(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, t):
        print(t)
        print(t.shape)
        return t

class CNN(nn.Module):
    """Convolutional Neural Networks"""
    def __init__(self, n_kernels, kernel_size, input_size, linear_size, output_horizon):
        super(CNN, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_kernels, kernel_size=kernel_size),
            Square(),
            #nn.AvgPool1d(kernel_size=2), 
            nn.Flatten(),         
            #nn.Linear(out_channels * floor((window_size - kernel_size + 1) / 2), 10),  #2 depends from kernel_size of avgpool layer
            nn.Linear(n_kernels * (input_size - kernel_size + 1), linear_size), #use without avgpool
            nn.Linear(linear_size, output_horizon)
        )

    def forward(self, x):
        out = self.main(x)
        return out


        