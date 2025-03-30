import torch
import torch.nn as nn

class FFNN(nn.Module):

    def __init__(self, input_size, n_classes):
        super().__init__()

        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, n_classes)
        self.dropout = nn.Dropout(p=0.2)
        self.soft_max = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.relu(self.linear3(x))
        x = self.dropout(x)
        x = self.relu(self.linear4(x)) 
        if not self.training: # Logits for training, probability for testing
            x = self.soft_max(x)
        
        return x