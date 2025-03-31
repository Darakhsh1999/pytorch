"""
Expert pytorch training template. For extracting maximal performance and advanced DL techniques
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
- Model weight initialization
- Single batch overfit (dev run)
- Automatic Mixed Precision (AMP)
- Gradient clipping
- Gradient accumulation
- Weight decey
- torch.compile optimization
"""
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Optimizer
from contextlib import nullcontext
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("..")
from utils.early_stoppage import EarlyStopping


class Parameters():

    # Training 
    n_epochs = 50
    batch_size = 64
    n_grad_accumulations = 16
    stopping_criterion = EarlyStopping(patience=5, mode="min") 

    # Hardware
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    # AMP
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16 # bfloat16'|'float16'
    ctx = nullcontext() if (device_name == 'cpu') else torch.autocast(device_type=device_name, dtype=dtype)
    gradient_clipping = 2.0
    scaler = torch.GradScaler(enabled=(dtype == torch.float16))


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

        # Weight initialization
        self.apply(self._init_weight)
    

    def _init_weight(self, module):
        """ Manually controlled weight initialization """
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        
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
    scheduler: ReduceLROnPlateau,
    overfit: bool = False 
    ):

    print(f"Started training on device: {p.device} | dtype: {p.dtype}")
    torch.set_float32_matmul_precision("high")
    
    train_pbar = tqdm(range(p.n_epochs), desc=f'Training', position=0)
    for epoch in train_pbar:

        # Train epoch
        model.train()
        train_loss = 0.0
        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc="[Step]", position=1, leave=False)):
            if overfit and (batch_idx > 0): break # when True, overfit single batch

            # Transfer data to GPU
            x, y = x.to(p.device), y.to(p.device)

            # Automatic Mixed Precision (AMP)
            with p.ctx:
                outputs = model(x)
                loss = loss_fn(outputs, y)

            loss /= p.n_grad_accumulations
            train_loss += loss.item()
            p.scaler.scale(loss).backward()

            if ((batch_idx+1) % p.n_grad_accumulations == 0) or overfit:

                # Gradient clipping
                if p.gradient_clipping != 0.0:
                    p.scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), p.gradient_clipping)

                # Optimizer step
                p.scaler.step(optimizer)
                p.scaler.update()
                optimizer.zero_grad()

        train_loss = train_loss/len(train_loader)
        
        # Validation
        val_loss = test(model, p, val_loader, loss_fn)

        # LR scheduler
        scheduler.step(val_loss)

        # Update tqdm description
        train_pbar.set_description_str(f"[Training] L_tr={train_loss:.4f}|L_val={val_loss:.4f}"+(f"| norm: {norm:.3f}" if p.gradient_clipping!=0 else ""))

        # Check early stoppage
        if p.stopping_criterion(model, val_loss):
            p.stopping_criterion.load_best_model(model)
            train_pbar.close()
            print(f"Stopped training early at epoch {epoch+1}, best score: {p.stopping_criterion.best_score:.4f}")
            return
    

    # Manually load best model at end
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

    compile_model = True
    save_model = False

    # Hyperparameter class
    p = Parameters()

    # Create dataset
    dataset = LinearDataset(num_samples=50000)
    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=p.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=p.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=p.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    # Create model
    model = MLP()
    model.to(p.device)
    if compile_model:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model, mode="default")

    # Loss & Optimizer & Scheduler
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1, fused=True)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5)
    
    # Train the model
    train(model, p, train_loader, val_loader, loss_fn, optimizer, scheduler)
    
    # Evaluate the model
    test_loss = test(model, p, test_loader, loss_fn)


    if save_model:
        model.to("cpu")
        torch.save(model, "model_expert.pt")
    