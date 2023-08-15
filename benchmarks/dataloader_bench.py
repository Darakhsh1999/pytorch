"""
Benchmarks for DataLoaders. Investigating the effect of varying the parameters;
- batch_size
- num_workers
- pin_memory

"""

if __name__ == "__main__":

    import sys
    import torch
    from tqdm import tqdm
    from time import time
    from torchvision import transforms
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader

    sys.path.append(".") # package directory
    sys.path.append("../") # direct
    from models.cnn import CNN

    # Bench parameters
    n_iterations = 3
    use_transformed_data = True
    test1 = False
    test2 = False
    test3 = True

    # Model
    model = CNN()
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    # Data
    if use_transformed_data:
        train_data = MNIST(root="../data", train=True, download=True, transform=transforms.ToTensor())
    else:
        train_data = MNIST(root="../data", train=True, download=True)

    # Different batch sizes
    train_loader_batch = {
        "B1": DataLoader(train_data, batch_size=1),
        "B32": DataLoader(train_data, batch_size=32),
        "B64": DataLoader(train_data, batch_size=64),
        "B128": DataLoader(train_data, batch_size=128)
    }

    # Memory pin (4 different batch sizes)
    train_loader_pin = {
        "B1P": DataLoader(train_data, batch_size=1, pin_memory=True),
        "B32P": DataLoader(train_data, batch_size=32, pin_memory=True),
        "B64P": DataLoader(train_data, batch_size=64, pin_memory=True),
        "B128P": DataLoader(train_data, batch_size=128, pin_memory=True),
    }

    # Different number of workers (2 different batch sizes)
    train_loader_worker1 = {
        "B1W0": DataLoader(train_data, batch_size=1, num_workers=0),
        "B1W1": DataLoader(train_data, batch_size=1, num_workers=1),
        "B1W2": DataLoader(train_data, batch_size=1, num_workers=2),
        "B1W4": DataLoader(train_data, batch_size=1, num_workers=4),
        "B1W8": DataLoader(train_data, batch_size=1, num_workers=8),
        "B1W16": DataLoader(train_data, batch_size=1, num_workers=16)
    }
    train_loader_worker64 = {
        "B64W0": DataLoader(train_data, batch_size=64, num_workers=0),
        "B64W1": DataLoader(train_data, batch_size=64, num_workers=1),
        "B64W2": DataLoader(train_data, batch_size=64, num_workers=2),
        "B64W4": DataLoader(train_data, batch_size=64, num_workers=4),
        "B64W8": DataLoader(train_data, batch_size=64, num_workers=8),
        "B64W16": DataLoader(train_data, batch_size=64, num_workers=16)
    }


    print("Starting benchmark...")

    ##### Looping through data #####
    if test1:
        print("Looping through data tests:\n")

        print("Batch size test")
        for bench_name, dataloader in train_loader_batch.items():
            t = time()
            for i in range(n_iterations):
                for (img, labels) in dataloader:
                    pass
            t = (time() - t) / n_iterations
            print(f"{bench_name:4} = {t:.3f} s/iter")
        print(20*"-")

        print("Pin memory test")
        for bench_name, dataloader in train_loader_pin.items():
            t = time()
            for i in range(n_iterations):
                for (img, labels) in dataloader:
                    pass
            t = (time() - t) / n_iterations
            print(f"{bench_name:4} = {t:.3f} s/iter")
        print(20*"-")


        print("Worker number test (B=1)")
        for bench_name, dataloader in train_loader_worker1.items():
            t = time()
            for i in range(n_iterations):
                for (img, labels) in dataloader:
                    pass
            t = (time() - t) / n_iterations
            print(f"{bench_name:5} = {t:.3f} s/iter")
        print(20*"-")

        print("Worker number test (B=64)")
        for bench_name, dataloader in train_loader_worker64.items():
            t = time()
            for i in range(n_iterations):
                for (img, labels) in dataloader:
                    pass
            t = (time() - t) / n_iterations
            print(f"{bench_name:6} = {t:.3f} s/iter")
        print(20*"-")


    ###### Predicting with model CPU #####
    if test2:
        print("Model prediction on CPU tests:\n")

        with torch.no_grad():

            print("Batch size test")
            for bench_name, dataloader in train_loader_batch.items():
                t = time()
                for i in range(n_iterations):
                    for (img, labels) in dataloader:
                        pred = model(img)
                t = (time() - t) / n_iterations
                print(f"{bench_name:4} = {t:.3f} s/iter")
            print(20*"-")

            print("Pin memory test")
            for bench_name, dataloader in train_loader_pin.items():
                t = time()
                for i in range(n_iterations):
                    for (img, labels) in dataloader:
                        pred = model(img)
                t = (time() - t) / n_iterations
                print(f"{bench_name:6} = {t:.3f} s/iter")
            print(20*"-")

            print("Worker number test (B=1)")
            for bench_name, dataloader in train_loader_worker1.items():
                t = time()
                for i in range(n_iterations):
                    for (img, labels) in dataloader:
                        pred = model(img)
                t = (time() - t) / n_iterations
                print(f"{bench_name:5} = {t:.3f} s/iter")
            print(20*"-")

            print("Worker number test (B=64)")
            for bench_name, dataloader in train_loader_worker64.items():
                t = time()
                for i in range(n_iterations):
                    for (img, labels) in dataloader:
                        pred = model(img)
                t = (time() - t) / n_iterations
                print(f"{bench_name:6} = {t:.3f} s/iter")
            print(20*"-")


    ###### Predicting with model GPU #####
    if test3:
        print("Model prediction on GPU tests:\n")

        model = model.to(device)
        with torch.no_grad():

            print("Batch size test")
            for bench_name, dataloader in train_loader_batch.items():
                t = time()
                for i in range(n_iterations):
                    for (img, labels) in dataloader:
                        img = img.to(device)
                        pred = model(img)
                t = (time() - t) / n_iterations
                print(f"{bench_name:4} = {t:.3f} s/iter")
            print(20*"-")

            print("Pin memory test")
            for bench_name, dataloader in train_loader_pin.items():
                t = time()
                for i in range(n_iterations):
                    for (img, labels) in dataloader:
                        img = img.to(device)
                        pred = model(img)
                t = (time() - t) / n_iterations
                print(f"{bench_name:6} = {t:.3f} s/iter")
            print(20*"-")

            print("Worker number test (B=1)")
            for bench_name, dataloader in train_loader_worker1.items():
                t = time()
                for i in range(n_iterations):
                    for (img, labels) in dataloader:
                        img = img.to(device)
                        pred = model(img)
                t = (time() - t) / n_iterations
                print(f"{bench_name:5} = {t:.3f} s/iter")
            print(20*"-")

            print("Worker number test (B=64)")
            for bench_name, dataloader in train_loader_worker64.items():
                t = time()
                for i in range(n_iterations):
                    for (img, labels) in dataloader:
                        img = img.to(device)
                        pred = model(img)
                t = (time() - t) / n_iterations
                print(f"{bench_name:6} = {t:.3f} s/iter")
            print(20*"-")
