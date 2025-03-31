"""
Advanced pytorch training template with profiling enabled.
- Parameter class
- Dummy data
- MLP model
- automatic device
- val/test function with @torch.no_grad decorator
- validation printout
- Early stopping
- LR scheduler
- Tqdm progress bar
- Dataloader optimization
- Model saving
- PyTorch profiling
"""
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Optimizer
from torch.profiler import profiler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("..")
from utils.early_stoppage import EarlyStopping


class Parameters():

    # Training 
    n_epochs = 50
    batch_size = 32
    stopping_criterion = EarlyStopping(patience=15, mode="min") 

    # Hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Profiler
    profiler_start_steps = 3 
    profiler_end_steps = 16
    profiler = profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA if (device == 'cuda') else None
        ],
        schedule=profiler.schedule(
            wait=profiler_start_steps,
            warmup=2,
            active=profiler_end_steps - profiler_start_steps,
            repeat=1
        ),
        on_trace_ready=profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )


class LinearDataset(Dataset):
    def __init__(self, num_samples=5000, noise_std=0.5):
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
    def __init__(self, input_dim=1, hidden_dims=[16, 32, 16], output_dim=1):
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
    scheduler: ReduceLROnPlateau 
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_pbar = tqdm(range(p.n_epochs), desc=f'Training', position=0)
    
    p.profiler.start()
    for epoch in train_pbar:

        # Train epoch
        model.train()
        train_loss = 0.0
        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc="[Step]", position=1, leave=False)):
            x, y = x.to(device), y.to(device)

            # Forward pass
            outputs = model(x)
            loss = loss_fn(outputs, y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            p.profiler.step()
        train_loss = train_loss/len(train_loader)
        
        # Validation
        val_loss = test(model, p, val_loader, loss_fn)

        # LR scheduler
        scheduler.step(val_loss)

        # Update tqdm description
        train_pbar.set_description_str(f"[Training] L_tr={train_loss:.4f}|L_val={val_loss:.4f}")

        # Check early stoppage
        if p.stopping_criterion(model, val_loss):
            p.profiler.stop()
            p.stopping_criterion.load_best_model(model)
            train_pbar.close()
            print(f"Stopped training early at epoch {epoch+1}, best score: {p.stopping_criterion.best_score:.4f}")
            return
    

    # Manually load best model at end
    p.profiler.stop()
    p.stopping_criterion.load_best_model(model)
    return
        

@torch.no_grad
def test(
    model: nn.Module,
    p: Parameters,
    data_loader: LinearDataset,
    loss_fn: _Loss
    ):

    model.eval()
    total_loss = 0
    for _, (x, y) in enumerate(tqdm(data_loader, desc="[Validation]", position=1, leave=False)):
        x, y = x.to(p.device), y.to(p.device)
        outputs = model(x)
        loss = loss_fn(outputs, y)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    return avg_loss





if __name__ == "__main__":

    save_model = False

    p = Parameters()

    # Create dataset
    dataset = LinearDataset(num_samples=1000, noise_std=0.5)
    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=p.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=p.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=p.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    # Create model
    model = MLP()
    model.to(p.device)

    # Loss & Optimizer & Scheduler
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5)
    
    # Train the model
    train(model, p, train_loader, val_loader, loss_fn, optimizer, scheduler)

    print(p.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    # Evaluate the model
    test_loss = test(model, p, test_loader, loss_fn)

    if save_model:
        model.to("cpu")
        torch.save(model, "model_advanced.pt")
    