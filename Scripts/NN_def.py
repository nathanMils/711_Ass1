import torch
import torch.nn as nn
from torch.utils.data import Dataset


class NathansWeirdNN(nn.Module):
    def __init__(self):
        super(NathansWeirdNN, self).__init__()
        self.l1 = nn.Linear(X_tensor.shape[1], 64)  # First fully connected hidden layer
        self.l2 = nn.Linear(64, 32)                 # Second fully connected hidden layer
        self.l3 = nn.Linear(32, 3)                  # Fully connected output layer (3 outputs for y)
        self.a3 = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l(x)
        x = self.a3(x)
        return x