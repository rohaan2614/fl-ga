import random
import os
import torch

import pandas as pd
from torch.nn import CrossEntropyLoss
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from torch.optim import SGD
# from torch.optim.lr_scheduler import StepLR
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from metric import Metric
from agent import Agent
from dataset_partitioner import DatasetPartitioner

from models.mnist_lenet import LeNet5
from utils import (set_flatten_model_back,
                   get_flatten_model_param, accuracy)


class Server:
    def __init__(self, *, model, criterion, device, lr=0.01):
        self.model = model.to(device)
        self.flatten_params = get_flatten_model_param(self.model).to(device)
        self.criterion = criterion
        self.device = device
        self.num_arb_participation = 0
        self.num_uni_participation = 0
        self.momentum = self.flatten_params.clone().zero_()
        self.lr = lr
        print('Using', device)

    # def avg_clients(self, clients: list[Agent]):
    #     self.flatten_params.zero_()
    #     for client in clients:
    #         self.flatten_params += get_flatten_model_param(
    #             client.model).to(self.device)
    #     self.flatten_params.div_(len(clients))
    #     set_flatten_model_back(self.model, self.flatten_params)

    def avg_clients(self, clients: list[Agent]):
        for client in clients:
            self.flatten_params -= get_flatten_model_param(
                client.model).to(self.device).mul_(self.lr / len(clients))
        set_flatten_model_back(self.model, self.flatten_params)

    def eval(self, test_dataloader) -> tuple[float, float]:
        self.model.eval()
        val_accuracy = Metric("val_accuracy")
        val_loss = Metric("val_loss")
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            val_accuracy.update(accuracy(outputs, targets).item())
            val_loss.update(self.criterion(outputs, targets).item())
        return val_loss.avg, val_accuracy.avg

    # Determine the sampling method by q
    def determine_sampling(self, q: float, sampling_type: float) -> float:
        if "_" in sampling_type:
            sampling_methods = sampling_type.split("_")
            if random.random() < q:
                self.num_uni_participation += 1
                return "uniform"
            else:
                self.num_arb_participation += 1
                return sampling_methods[1]
        else:
            return sampling_type

    def get_num_uni_participation(self) -> int:
        return self.num_uni_participation

    def get_num_arb_participation(self) -> int:
        return self.num_arb_participation


def local_update_fedavg(clients: list[Agent], server, local_steps):
    train_loss_sum, train_acc_sum = 0, 0
    for client in clients:
        train_loss, train_acc = client.train_k_step_fedavg(k=local_steps)
        train_loss_sum += train_loss
        train_acc_sum += train_acc
    return train_loss_sum / len(clients), train_acc_sum / len(clients)


if __name__ == '__main__':
    # set up
    model = LeNet5()
    criterion = CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_clients = 5
    batch_size = 128
    rounds = 600
    lr = 0.01
    local_steps = 1
    evaluation_interval = 10
    rounds_list = [0]

    # Load datasets
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize the image to 32x32
        transforms.ToTensor(),        # Convert the image to a tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize the image
    ])
    print("Load data sets...\n\t-> Train  ", end='', flush=True)
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    print("OK\n\t-> Test   ", end='', flush=True)
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    print("OK")

    print("Split dataset...")
    train_dataset_partition = DatasetPartitioner(
        targets=train_dataset.targets,
        num_clients=num_clients,
        seed=42)

    print("Generating client agents:")
    clients = []
    for i in range(num_clients):
        print(f'\t-> Agent {i+1}...', end='', flush=True)

        # Get the indices for the ith client
        indices = train_dataset_partition[i]
        subset = Subset(train_dataset, indices)
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        # client
        model = LeNet5()
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
        client = Agent(model=model,
                       optimizer=optimizer,
                       #    scheduler=StepLR(optimizer, step_size=10, gamma=0.1),
                       criterion=CrossEntropyLoss(),
                       train_loader=train_loader,
                       device=device)
        clients.append(client)

        print('OK')

    # Initialize the server
    server = Server(model=LeNet5(),
                    criterion=CrossEntropyLoss(),
                    device=device)

#     writer = SummaryWriter(
#     os.path.join(
#         "output",
#         "mnist",
#         f"clients={num_clients},rounds={rounds},lr={lr},",
    #     )
    # )

    loss = [0]
    accuracies = [0]
    for rnd in tqdm(range(1, 1 + rounds), desc="Training Progress"):
        # Train
        [client.pull_model_from_server(server) for client in clients]
        train_loss, train_acc = local_update_fedavg(clients=clients,
                                                    server=server,
                                                    local_steps=local_steps)

        # Broadcast
        server.avg_clients(clients)

        # Evaluate
        if rnd % evaluation_interval == 0:
            eval_loss, eval_acc = server.eval(test_loader)
            print(
                f'Round: {rnd}/{rounds}, Accuracy: {eval_acc}, Loss: {eval_loss}')
            loss.append(eval_loss)
            accuracies.append(eval_acc)
            rounds_list.append(rnd)

    print(f'lengths:\n\t-> Round: ', end='', flush=True)
    print(f'{len(rounds_list)}\n\t-> Loss: ', end='', flush=True)
    print(f'{len(loss)}\n\t-> Accuracy: ', end='', flush=True)
    print(len(accuracies))

    df = pd.DataFrame(data={
        'Round': rounds_list,
        'Test Loss': loss,
        'Accuracy': accuracies
    })

    df.to_csv(f'server_results_{rounds}_rounds_{num_clients}_clients.csv',
              index=False)
