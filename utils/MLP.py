import torch
import torch.nn as nn
import torch.nn.functional as f


class MLP(nn.Module):
    def __init__(self, data_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(data_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        a = self.fc1(x)
        a = f.relu(a)

        b = self.fc2(a)
        b = f.relu(b)

        c = self.fc3(b)

        return c
