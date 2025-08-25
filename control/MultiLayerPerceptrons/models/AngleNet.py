import torch
# import torchvision
# import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np

# Angle-based control

class AngleNet(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layer_size: list):
        super().__init__()
        self.net1 = torch.nn.Linear(input_dim,hidden_layer_size[0])
        self.net2 = torch.nn.Linear(hidden_layer_size[0], hidden_layer_size[1])
        self.net3 = torch.nn.Linear(hidden_layer_size[1], output_dim)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.net1(x)
        x = torch.relu(x)
        x = self.net2(x)
        x = torch.relu(x)
        x = self.net3(x)
        x = torch.sigmoid(x)
        return x
    
class DutyRatioNet(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layer_size: list):
        super().__init__()
        self.net1 = torch.nn.Linear(input_dim,hidden_layer_size[0])
        self.net2 = torch.nn.Linear(hidden_layer_size[0], hidden_layer_size[1])
        self.net3 = torch.nn.Linear(hidden_layer_size[1], hidden_layer_size[2])
        self.net4 = torch.nn.Linear(hidden_layer_size[2], output_dim)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.net1(x)
        x = torch.relu(x)
        x = self.net2(x)
        x = torch.relu(x)
        x = self.net3(x)
        x = torch.relu(x)
        x = self.net4(x)
        x = torch.relu(x)
        return x
    
class AngleTimeNet(torch.nn.Module):
    # predict duty ratio percentage and and its duration from target and current angles
    def __init__(self, input_dim: int, output_dim: int, hidden_layer_size: list):
        super().__init__()
        self.num_hiden_layers = len(hidden_layer_size)