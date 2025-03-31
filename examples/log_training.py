"""
Advanced pytorch training template with MLFlow logging added.
Run: "mlflow ui --backend-store-uri file:./mlruns" to spin up UI server for watching logs
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
- MLflow logging 
"""
import os.path as osp
import torch
import mlflow
import mlflow.pytorch
import tempfile
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Optimizer
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
    lr = 0.01
    stopping_criterion = EarlyStopping(patience=20, mode="min") 

    # Hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        train_loss = train_loss/len(train_loader)
        
        # Validation
        val_loss = test(model, p, val_loader, loss_fn)

        # Log loss metrics
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        # LR scheduler
        scheduler.step(val_loss)

        # Update tqdm description
        train_pbar.set_description_str(f"[Training] L_tr={train_loss:.4f}|L_val={val_loss:.4f}")

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

    save_model = False

    # Set experiment name
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("log_training tests")
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    p = Parameters()
    param_dict =  {k:v for (k,v) in p.__class__.__dict__.items() if (not k.startswith("__"))}

    # Create dataset
    dataset = LinearDataset(num_samples=50000, noise_std=0.5)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=p.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5)
    
    # Train the model
    with mlflow.start_run():
        mlflow.set_tag("model class", model.__call__.__name__)

        # Log parameters
        param_dict.update({
            "model_name": model.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__,
            "loss_fn": loss_fn.__class__.__name__}
        )
        mlflow.log_params(param_dict)

        # Log artifacts of model and optimizer (alternative 1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(osp.join(tmp_dir,"optimizer.txt"), "w") as f:
                f.write(str(optimizer))
            with open(osp.join(tmp_dir,"model_summary.txt"), "w") as f:
                f.write(str(model))
            mlflow.log_artifacts(tmp_dir, "model_optimizer")

        # Log artifacts of model and optimizer (alternative 2)
        mlflow.pytorch.log_model(model, "modelv2")

        # Start model training
        train(model, p, train_loader, val_loader, loss_fn, optimizer, scheduler)
    
        # Evaluate the model
        test_loss = test(model, p, test_loader, loss_fn)
        mlflow.log_metric("test_loss", test_loss)

    if save_model:
        model.to("cpu")
        torch.save(model, "model_advanced.pt")

    print(f"To view the MLflow UI, run 'mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}'")
    