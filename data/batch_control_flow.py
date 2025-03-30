import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class InvertTransform:

    def __call__(self, sample):
        x, y = sample
        if int(x) % 2 == 0:
            y *= -1
        return x ,y

class MultiplyTransform:

    def __call__(self, sample):
        x, y = sample
        if int(x) % 2 == 1:
            y *= 1.5
        return x ,y


class CustomData(Dataset):

    def __init__(self, transform=None, n_samples=100, k=3.0, m=2.0):

        self.transform = transform
        self.n_samples = n_samples
        self.k = k 
        self.m = m 
        self.x = torch.arange(n_samples, dtype=torch.float32)
        self.y = k * self.x + m

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.x[idx], self.y[idx]
        if self.transform:
            return self.transform(sample)
        else:
            return sample
    


if __name__ == "__main__":

    transform = InvertTransform()
    transform_compose = transforms.Compose(
        [
        InvertTransform(),
        MultiplyTransform()
        ])

    data = CustomData(transform=transform, n_samples=10)
    dataloader = DataLoader(data, batch_size=16)

    batch_iter = iter(dataloader)
    batch = next(batch_iter)

    for batch in dataloader:
        x, y = batch
        print(f"x = {x}, y = {y}")


