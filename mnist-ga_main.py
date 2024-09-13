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
import torch.optim as optim
from tqdm import tqdm
import pickle

from agent import Agent
from dataset_partitioner import DatasetPartitioner
from server import Server
from models.cnn_mnist import CNN_Mnist as Model
from utils import local_train_1_step
# , local_update_fedavg_ga

import ga_utils
from ga import GA

if __name__ == '__main__':
    model_name = 'mnist'
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Train a model and optionally append a job_id to the CSV output file.')
    parser.add_argument('--job_id', type=str, default='',
                        help='Optional job ID to append to the CSV file.')
    args = parser.parse_args()

    # set up
    start_time = time.time()
    model = Model()
    criterion = CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_clients = 1
    batch_size = 128
    rounds = 1000
    q = 10
    lr = 0.01
    local_steps = 1
    evaluation_interval = 10
    rounds_list = [0]

    # Load datasets
    transform = transforms.Compose([
        # transforms.Resize((32, 32)),  # Resize images to 32x32
        transforms.ToTensor(),
        # Normalize using MNIST statistics
        transforms.Normalize((0.1307,), (0.3081,))
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
        model = Model()
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.5,
        )
        client = Agent(model=model,
                       optimizer=optimizer,
                       criterion=CrossEntropyLoss(),
                       train_loader=train_loader,
                       device=device)
        clients.append(client)
        print('OK')

    # Initialize the server
    server = Server(model=Model(),
                    criterion=CrossEntropyLoss(),
                    device=device,
                    lr=0.1)

    loss = [0]
    accuracies = [0]

    weights, updated_weights, local_updates, gws, normalized_gws = [], [], [], [], []
    w_ts = []
    wt_plus_1s = []
    post_agg_wts = []

    for rnd in tqdm(range(rounds), desc="Training Progress"):
        w_ts.append([])
        wt_plus_1s.append([])
        post_agg_wts.append([])
        local_updates.append([])
        normalized_gws.append([])
        gws.append([])
        shapes = None

        # Train
        [client.pull_model_from_server(server) for client in clients]

        train_loss, train_acc = local_train_1_step(clients=clients)

        for i, client in enumerate(clients):
            local_update = client.model_grad
            local_updates[rnd].append(local_update)
            
            # print('local_update type:', type(local_update))

            seed = random.randint(0, 100)
            ga = GA(seed=seed,
                    d=len(local_update),
                    q=int(q),
                    device=device)
            w = ga.w(delta=local_update)
            gw = ga.delta(w=w)

            gws[rnd].append(gw.cpu())
            
            client_weights, shapes = ga_utils.flatten_vector(client.model.state_dict(), device=device)

            # gw_normalized = ga_utils.scale_to_match_range(delta=local_update,
            #                                               gw=gw)

            # normalized_gws[rnd].append(gw_normalized.cpu())

            new_weights = client_weights - (local_update*server.lr)/len(clients)
            client.model.load_state_dict(
                ga_utils.vector_to_state_dict(new_weights, shapes))

        # Aggregate
        server.avg_clients(clients)

        for client in clients:
            post_agg_wt = {name: weights.clone().detach().cpu()
                           for name, weights in client.model.state_dict().items()}
            post_agg_wt_flat = ga_utils.flatten_vector(
                state_dict=post_agg_wt, device=device)[0]
            post_agg_wts[rnd].append(post_agg_wt_flat.cpu())

        # Evaluate
        if rnd % evaluation_interval == 0:
            eval_loss, eval_acc = server.eval(test_loader)
            print(
                f'Round: {rnd + 1}/{rounds}, Accuracy: {eval_acc}, Loss: {eval_loss}')
            loss.append(eval_loss)
            accuracies.append(eval_acc)
            rounds_list.append(rnd)

    df = pd.DataFrame(data={
        'Round': rounds_list,
        'Test Loss': loss,
        'Accuracy': accuracies
    })

    # Generate CSV filename with optional job_id
    csv_filename = f'{model_name}_{rounds}-rounds_{num_clients}-clients_{q}-vectors'
    if args.job_id:
        csv_filename += f'_{args.job_id}'
    csv_filename += '.csv'

    df.to_csv(csv_filename, index=False)

    with open(f'{args.job_id}.pkl', 'wb') as f:
        data = {
            'w_ts': w_ts,
            'wt_plus_1s': wt_plus_1s,
            'post_agg_wts': post_agg_wts,
            'local_updates': local_updates,
            'normalized_gws': normalized_gws,
            'gws': gws
        }
        pickle.dump(data, f)

    print("Execution Time:", time.time() - start_time)
