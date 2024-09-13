import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from metric import Metric
from utils import set_flatten_model_back, accuracy
from models.mnist_lenet import LeNet5
from utils import get_flatten_model_param, get_flatten_model_grad
    
class Agent:
    def __init__(self, *, model, optimizer, criterion, train_loader, device, scheduler= None):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.train_loss = Metric("train_loss")
        self.train_accuracy = Metric("train_accuracy")
        self.device = device
        self.batch_idx = 0
        self.epoch = 0
        self.data_generator = self.get_one_train_batch()
        self.model_grad = torch.zeros_like(get_flatten_model_param(self.model))

    # lazy loading
    def get_one_train_batch(self):
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # return 1 batch at a time i.e generator aspect
            yield batch_idx, (inputs, targets)

    def reset_epoch(self):
        self.data_generator = self.get_one_train_batch()
        self.batch_idx = 0
        self.epoch += 1
        self.train_loss = Metric("train_loss")
        self.train_accuracy = Metric("train_accuracy")

    def pull_model_from_server(self, server):
        if self.device != "cpu":
            # Notice the device between server and client may be different.
            with torch.device(self.device):
                # This context manager is necessary for the clone operation.
                set_flatten_model_back(
                    self.model, server.flatten_params.to(self.device)
                )
        else:
            set_flatten_model_back(self.model, server.flatten_params)

    def decay_lr_in_optimizer(self, gamma: float):
        for g in self.optimizer.param_groups:
            g["lr"] *= gamma

    def train_k_step_fedavg(self, k: int):
        self.model.train()
        self.model_grad = torch.zeros_like(get_flatten_model_param(self.model))
        for i in range(k):
            try:
                batch_idx, (inputs, targets) = next(self.data_generator)
            except StopIteration:
                loss, acc = self.train_loss.avg, self.train_accuracy.avg
                self.reset_epoch()
                return loss, acc
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.model_grad += get_flatten_model_grad(self.model)
            self.optimizer.step()
            self.train_loss.update(loss.item())
            self.train_accuracy.update(accuracy(outputs, targets).item())
        return self.train_loss.avg, self.train_accuracy.avg
    
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
    
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize the image to 32x32
        transforms.ToTensor(),        # Convert the image to a tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize the image
    ])

    print("Data Loaders...\n\t-> Train  ", end='', flush=True)
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print("OK\n\t-> Test   ", end='', flush=True)

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    print("OK")
    

    model = LeNet5()
    # to Measure the model’s prediction error
    criterion = nn.CrossEntropyLoss()
    # to updates the model’s parameters to minimize the error
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # to adjust learning rate during training to improve convergence.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    
    agent = Agent(model=model,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  criterion=criterion,
                  train_loader=train_loader,
                  device=device)
    
    num_steps = 100
    for local_step in range(num_steps):
        print(f'Step {local_step + 1}/{num_steps}:', end=' ', flush=True)
        
        # Training using FedAvg
        train_loss, train_accuracy = agent.train_k_step_fedavg(k=1)
        print(f'Training loss: {train_loss:.4f}, Training accuracy: {train_accuracy:.4f}')
    