import random
import os
import torch
import argparse
import time

import pandas as pd
from torch.nn import CrossEntropyLoss
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from torch.optim import SGD
from tqdm import tqdm

from metric import Metric
from agent import Agent
from dataset_partitioner import DatasetPartitioner
from server import Server
from models.mnist_lenet import LeNet5
from utils import (set_flatten_model_back,
                   get_flatten_model_param,
                   local_update_fedavg, accuracy)

if __name__ == '__main__':
    model_name = 'mnist'
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train a model and optionally append a job_id to the CSV output file.')
    parser.add_argument('--job_id', type=str, default='', help='Optional job ID to append to the CSV file.')
    args = parser.parse_args()

    # set up
    start_time = time.time()
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
                       criterion=CrossEntropyLoss(),
                       train_loader=train_loader,
                       device=device)
        clients.append(client)
        print('OK')

    # Initialize the server
    server = Server(model=LeNet5(),
                    criterion=CrossEntropyLoss(),
                    device=device)

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

    df = pd.DataFrame(data={
        'Round': rounds_list,
        'Test Loss': loss,
        'Accuracy': accuracies
    })
    
     # Generate CSV filename with optional job_id
    csv_filename = f'{model_name}_{rounds}_rounds_{num_clients}_clients'
    if args.job_id:
        csv_filename += f'_{args.job_id}'
    csv_filename += '.csv'

    df.to_csv(csv_filename, index=False)
    
    print("Execution Time:", time.time() - start_time)
    
