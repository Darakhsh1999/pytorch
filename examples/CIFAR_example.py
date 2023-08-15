import sys
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset

sys.path.append(".") # package directory
from models.cnn import CNN

if __name__ == "__main__":

    # Hyperparameters
    batch_size = 64
    n_epochs = 30
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CNN model
    model = CNN(n_channels=3, linear_in=8)
    model = model.to(device)

    # Data sets
    train_data = CIFAR10(root='./data/datasets', train=True, download=True, transform=transform)
    test_data = CIFAR10(root='./data/datasets', train=False, download=True, transform=transform)
    visual_data = CIFAR10(root='./data/datasets', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4) 
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4) 
    idx_to_classes = test_data.classes

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
            img = img.to(device) # (N,3,H,W), float32
            labels = labels.to(device) # (N,), int64

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
    visual_subset = Subset(visual_data, indices=im_idx)

    model = model.to("cpu")
    model.eval()
    with torch.no_grad():
        for idx, (img, label) in enumerate(test_subset):

            probability =  model(img[None])
            class_prediction = torch.argmax(probability, dim=-1).flatten()
            class_name = idx_to_classes[class_prediction.item()]
            target_name = idx_to_classes[label]

            np_image = np.moveaxis(visual_subset[idx][0].numpy().squeeze(), 0, -1)
            plt.imshow(np_image)
            plt.title(f"Prediction = {class_name}, Target = {target_name}")
            plt.show()
