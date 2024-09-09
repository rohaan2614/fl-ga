import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import Counter


class DatasetPartitioner:
    def __init__(self, *, targets, num_clients, seed=42, balance=True):
        self.targets = targets
        self.num_clients = num_clients
        self.seed = seed
        self.balance = balance
        self.partitions = self.partition_data()

    def partition_data(self):
        np.random.seed(self.seed)
        indices = np.arange(len(self.targets))
        np.random.shuffle(indices)

        # Get class-wise indices if balance=True
        if self.balance:
            class_indices = {}
            for i, target in enumerate(self.targets):
                if target.item() not in class_indices:
                    class_indices[target.item()] = []
                class_indices[target.item()].append(i)
            # Partition class-wise indices into clients
            partitions = [[] for _ in range(self.num_clients)]
            for class_idx in class_indices:
                class_data = class_indices[class_idx]
                class_data = np.array_split(class_data, self.num_clients)
                for i in range(self.num_clients):
                    partitions[i].extend(class_data[i])
        else:
            partitions = np.array_split(indices, self.num_clients)

        # Convert partitions to a list of PyTorch tensors
        return [torch.tensor(partition, dtype=torch.long) for partition in partitions]

    def __getitem__(self, index):
        return self.partitions[index]

    def __len__(self):
        return len(self.partitions)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize the image to 32x32
        transforms.ToTensor(),        # Convert the image to a tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize the image
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)

    sample_indices = np.random.choice(len(train_dataset), 50, replace=False)
    train_dataset_sampled = Subset(train_dataset, sample_indices)

    NUM_CLIENTS = 2
    
    train_dataset_partition = DatasetPartitioner(
        targets=[train_dataset.targets[i] for i in sample_indices],
        num_clients=NUM_CLIENTS,
        seed=42
    )

    for idx in range(NUM_CLIENTS):
        client_data_indices = train_dataset_partition[idx]
        client_targets = [train_dataset.targets[i].item() for i in client_data_indices]

        # Count occurrences of each target label
        target_counts = Counter(client_targets)

        print(f"Client {idx+1}:")
        print(f"  -> Data indices = {client_data_indices.tolist()}")
        print(f"  -> Count of targets = {dict(target_counts)}\n")
    