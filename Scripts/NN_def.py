import torch
import torch.nn as nn
from torch.utils.data import Dataset


class NathansWeirdNN(nn.Module):
    def __init__(self):
        super(NathansWeirdNN, self).__init__()
        self.fc1 = nn.Linear(X_tensor.shape[1], 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 32)                 # Second hidden layer
        self.fc3 = nn.Linear(32, 3)                  # Output layer (3 classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x