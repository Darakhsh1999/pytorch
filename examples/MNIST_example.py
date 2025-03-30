import sys
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset

sys.path.append(".") # package directory
sys.path.append("../") # direct
from models.cnn import CNN

# Hyperparameters
batch_size = 64
lr = 0.001
n_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

# CNN model
model = CNN()
model = model.to(device)

# Data sets
train_data = MNIST(root="./data/datasets", train=True, download=True, transform=transforms.ToTensor())
test_data = MNIST(root="./data/datasets", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Optimizer and loss function 
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

### Train model 
model.train()
for epoch_idx in range(n_epochs):

    # Loop through training data
    epoch_loss = 0.0
    for img, labels in train_loader:
        
        # Load in batch and cast image to float32
        img = img.to(device) # (N,1,H,W)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(img) # (N,10), float32
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    epoch_loss /= len(train_loader.dataset)
    print(f"Epoch {1+epoch_idx} loss = {epoch_loss:.4f}")


### Test evaluation
model.eval()
n_correct_predictions = 0.0
with torch.no_grad():
    for img, labels in test_loader:

        img, labels = img.to(device), labels.to(device)
        output_probability = model(img) # (N,10)

        predicted_batch_class = torch.argmax(output_probability, dim=-1) # (N,) class 0-9

        n_correct_predictions += (predicted_batch_class == labels).sum().cpu().item()

accuracy = n_correct_predictions / len(test_data)
print(f"Test accuracy = {accuracy*100:.2f}%")

### Test on random images
n_predictions = 10
im_idx = np.random.choice(len(test_data), n_predictions, replace=False)
test_subset = Subset(test_data, indices=im_idx)

model = model.to("cpu")
model.eval()
with torch.no_grad():
    for img, label in test_subset:

        probability =  model(img[None])
        class_prediction = torch.argmax(probability, dim=-1).flatten()

        plt.imshow(img.numpy().squeeze(), cmap="gray")
        plt.title(f"Prediction = {class_prediction.item()}, Target = {label}")
        plt.show()

