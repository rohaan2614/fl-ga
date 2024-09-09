import torch.nn as nn
import torch
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=120*1*1, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)  
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = torch.tanh(x)  
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
        x = torch.tanh(x)  
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = torch.tanh(x)  
        x = self.fc2(x)
        
        return x