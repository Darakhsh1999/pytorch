import sys
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

sys.path.append(".") # package directory
sys.path.append("../") # direct
from models.cnn import CNN

if __name__ == "__main__":

    # Data sets
    train_data = MNIST(root="../data", train=True, download=True, transform=transforms.ToTensor())
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    # Hyperparameters
    batch_size = 64
    lr = 0.001
    n_epochs = 20

    # Train data
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    #train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # CNN model
    model = CNN()
    model = model.to(device)

    # Optimizer and loss function 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    ### Train model 
    model.train()
    t = time()
    for epoch_idx in tqdm(range(n_epochs)):

        # Loop through training data
        epoch_loss = 0.0
        for img, labels in train_loader:
            
            # Load in batch and cast image to float32
            img = img.to(device) # (N,1,H,W)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(img) # (N,10)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= len(train_loader.dataset)
    t = (time() - t)/n_epochs
    print(f"t = {t:.2f} s/epoch")