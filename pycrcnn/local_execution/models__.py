import torch.nn as nn
import torch

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

class CNN(nn.Module):
    """Convolutional Neural Networks"""
    def __init__(self, input_size, hidden_dim, output_size):
        super(CNN, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_dim, kernel_size=1),
            #nn.ReLU(),
            Square(),
            #Cube(),
            nn.Flatten(),

            nn.Linear(hidden_dim, 10),
            nn.Linear(10, output_size)
        )

    def forward(self, x):
        out = self.main(x)
        return out


        