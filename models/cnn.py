import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, n_channels: int = 1, linear_in: int = 7):
        super().__init__()

        self.conv1 = nn.Conv2d(n_channels, 4, kernel_size=(3,3), padding="same")
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(3,3), padding="same")
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(3,3), padding="same")
        self.conv4 = nn.Conv2d(16, 32, kernel_size=(3,3), padding="same")
        self.linear1 = nn.Linear(32*linear_in*linear_in, 100)
        self.linear2 = nn.Linear(100, 10)
        self.soft_max = nn.Softmax(dim=-1)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.linear1(x)
        x = self.linear2(x)
        if not self.training: # Logits for BCE when training
            x = self.soft_max(x)
        return x