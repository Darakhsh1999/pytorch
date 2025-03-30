import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Parameters():

    # Training 
    n_epochs = 100
    batch_size = 64

    # Hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom dataset for y = 3x + 2 with Gaussian noise
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
def train(model, p, train_loader, val_loader, loss_fn, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    
    for epoch in range(p.n_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            outputs = model(x)
            loss = loss_fn(outputs, y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        val_loss = test(model, val_loader, loss_fn)
            
    return
        

# Evaluation function
def test(model, data_loader, loss_fn):
    device = model.device
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f'Loss: {avg_loss:.4f}')
    return avg_loss

if __name__ == "__main__":

    p = Parameters()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Train the model
    losses = train(model, p, train_loader, val_loader, loss_fn, optimizer, device, epochs=100)
    
    # Evaluate the model
    test_loss = test(model, test_loader, device)
    
    # Save the model
    torch.save(model.state_dict(), 'mlp_model.pth')
    print("Model saved to 'mlp_model.pth'")
    
    # Sample prediction
    sample_x = torch.tensor([[5.0]]).to(device)
    with torch.no_grad():
        sample_pred = model(sample_x).item()
    
    print(f"Prediction for x=5: {sample_pred:.4f} (True value without noise: {3*5+2})")