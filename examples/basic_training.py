"""
Most basic pytorch training template.
- Parameter class
- Dummy data
- MLP model
- automatic device
- val/test function
- validation printout
"""
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader

class Parameters():

    # Training 
    n_epochs = 30
    batch_size = 64
    lr = 0.01

    # Hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LinearDataset(Dataset):
    def __init__(self, num_samples=1000, noise_std=0.5):
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.x = torch.linspace(-10, 10, num_samples).view(-1, 1)
        self.y = 3 * self.x + 2 + torch.randn(num_samples, 1) * noise_std
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[16, 8], output_dim=1):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers with ReLU activations
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


# Training function
def train(
    model: nn.Module,
    p: Parameters,
    train_loader: LinearDataset,
    val_loader: LinearDataset,
    loss_fn: _Loss,
    optimizer: Optimizer,
    verbose: bool = True
    ):

    model.train()
    for epoch in range(p.n_epochs):
        for x, y in train_loader:
            x, y = x.to(p.device), y.to(p.device)
            
            # Forward pass
            outputs = model(x)
            loss = loss_fn(outputs, y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        val_loss = test(model, p, val_loader, loss_fn)
        if verbose: print(f"Epoch {epoch+1} had validation loss: {val_loss:.5f}")
            
    return
        

# Evaluation function
def test(
    model: nn.Module,
    p: Parameters,
    data_loader: LinearDataset,
    loss_fn: _Loss
    ):

    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(p.device), y.to(p.device)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    return avg_loss

if __name__ == "__main__":

    p = Parameters()

    # Create dataset
    dataset = LinearDataset(num_samples=1000, noise_std=0.5)
    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=p.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=p.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=p.batch_size, shuffle=False)
    
    # Create model
    model = MLP()
    model.to(p.device)

    # Loss
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=p.lr)
    
    # Train the model
    train(model, p, train_loader, val_loader, loss_fn, optimizer)
    
    # Evaluate the model
    test_loss = test(model, p, test_loader, loss_fn)
    